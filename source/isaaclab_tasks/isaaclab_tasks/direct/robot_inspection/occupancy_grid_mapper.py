import numpy as np
import warp as wp
import open3d as o3d
from scipy.spatial.transform import Rotation

from .warp_occupancy_fast import (
    update_occupancy_fast, 
    mark_visible_voxels, 
    update_visitation_kernel,
    reset_maps_kernel, 
    extract_local_maps_kernel,
    clamp_map_values, 
)


class InteractiveVoxelVisualizer:
    """Manages an interactive Open3D visualization window."""
    def __init__(self, voxel_size: float, title="Voxel Grid"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title)

        self.voxel_size = voxel_size
        self.grid = o3d.geometry.VoxelGrid()
        self._initialized = False
        self.robot_pose_marker = None

    def update_grid(self, points, colors):
        """Updates the point cloud geometry in the visualizer."""
        if points.shape[0] == 0:
            # Clear the geometry if there are no points
            if self._initialized:
                self.grid.clear()
                self.vis.update_geometry(self.grid)
            return
        temp_pcd = o3d.geometry.PointCloud()
        try:
            temp_pcd.points = o3d.utility.Vector3dVector(points)
            temp_pcd.colors = o3d.utility.Vector3dVector(colors)
            new_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(temp_pcd, voxel_size=self.voxel_size)
        except Exception as e:
            print(f"Error creating voxel grid: {e}")
            return

        if not self._initialized:
            self.grid = new_grid
            self.vis.add_geometry(self.grid)
            self._initialized = True

            view_ctl = self.vis.get_view_control()
            view_ctl.set_lookat([0.0, 0.0, 0.3])  # Point the camera at the center of your map
            view_ctl.set_front([-2, -1, 0.8]) # Set the camera's position (where it's looking from)
            view_ctl.set_up([0, 0, 1])     # Define the "up" direction (Z-axis is up)
            view_ctl.set_zoom(0.1)

        else:
            self.vis.remove_geometry(self.grid, reset_bounding_box=False)
            self.grid = new_grid # Combine the new grid into our existing one
            self.vis.add_geometry(self.grid, reset_bounding_box=False)
        # Process events to keep the window responsive
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def update_robot_pose(self, position: np.ndarray, orientation_quat: np.ndarray):
        """Creates or updates a coordinate frame representing the robot's pose."""
        # Create the coordinate frame geometry
        # The size parameter controls how large the axis marker is
        pose_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # Create a 4x4 transformation matrix from the position and quaternion
        transform_matrix = np.eye(4)
        
        # Isaac Sim uses (w, x, y, z), while SciPy uses (x, y, z, w)
        # We need to convert from Isaac Sim's convention to SciPy's
        w, x, y, z = orientation_quat
        scipy_quat = [x, y, z, w]
        
        # Set the rotation part of the matrix
        transform_matrix[:3, :3] = Rotation.from_quat(scipy_quat).as_matrix()
        # Set the translation part of the matrix
        transform_matrix[:3, 3] = position
        
        # Apply the transformation to the coordinate frame
        pose_marker.transform(transform_matrix)
        
        # If a marker already exists, remove it first
        if self.robot_pose_marker is not None:
            self.vis.remove_geometry(self.robot_pose_marker, reset_bounding_box=False)
        
        # Add the new, transformed marker to the visualizer
        self.vis.add_geometry(pose_marker, reset_bounding_box=False)
        self.robot_pose_marker = pose_marker

    def close(self):
        """Closes the visualization window."""
        self.vis.destroy_window()

class OccupancyGridMapper:
    """
    A class to manage and update a 3D occupancy grid using GPU-accelerated
    fast voxel traversal with NVIDIA Warp.
    """
    def __init__(
                self,
                num_envs,
                map_bounds: dict,
                resolution  :float,
                env_origins : np.array,
                visibility_surface_hits_only : bool = False,
                visualize_env_id: int | None = None,
                device="cuda"):
        """
        Initializes the occupancy grid mapper.

        Args:
            map_bounds (dict): A dictionary with keys {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}.
            voxel_size (float): The size of each voxel in meters.
            resolution (float): The size of each voxel in meters.
            visualize_env_id (int | None): The ID of the environment to visualize.
            device (str): The compute device for Warp ("cuda" or "cpu").
        """
        self.env_origins = env_origins
        self.num_envs = num_envs
        self.device = device
        self.map_bounds = map_bounds
        self.resolution = resolution
        self.visibility_surface_hits_only = visibility_surface_hits_only

        self.map_origin = np.array([
            self.map_bounds['x_min'],
            self.map_bounds['y_min'],
            self.map_bounds['z_min']
        ])
        self.world_map_origins = self.env_origins + self.map_origin
         # Call this after setting map_bounds and resolution
        self.initialize_map_dimensions()
        
        # Log-odds parameters
        self.log_odds_free = -0.4  # P(free) = 0.4
        self.log_odds_occupied = 0.8 # P(occupied) = 0.7
        self.log_odds_neutral = 0.0
        self.clamp_min = -5.0      # Min log-odds
        self.clamp_max = 5.0       # Max log-odds

        self.log_odds_visible = 0.4
        
        self.visualizer = None
        self.vis_env_id = visualize_env_id
        if self.vis_env_id is not None:
            if self.vis_env_id >= self.num_envs:
                print(f"Warning: visualize_env_id {self.vis_env_id} is out of bounds. Disabling visualization.")
                self.vis_env_id = None
            else:
                self.visualizer = InteractiveVoxelVisualizer(voxel_size=self.voxel_size, title=f"Voxel Grid (Env {self.vis_env_id})")
                print(f"Visualization enabled for environment {self.vis_env_id}.")

    def initialize_map_dimensions(self):
        self.x_width = int(np.ceil((self.map_bounds['x_max'] - self.map_bounds['x_min']) / self.resolution))
        self.y_width = int(np.ceil((self.map_bounds['y_max'] - self.map_bounds['y_min']) / self.resolution))
        self.z_width = int(np.ceil((self.map_bounds['z_max'] - self.map_bounds['z_min']) / self.resolution))
        self.map_dims = (self.x_width, self.y_width, self.z_width)
        # voxel size in meters
        self.voxel_size = self.resolution
        self.num_voxels_per_map = int(np.prod(self.map_dims))
        total_voxels = self.num_envs * self.num_voxels_per_map
        self.occupancy_map = wp.zeros(total_voxels, dtype=float, device=self.device)
        self.visibility_map = wp.zeros(total_voxels, dtype=float, device=self.device)
        self.visitation_map = wp.zeros(total_voxels, dtype=float, device=self.device)



        print(f"Initialized {self.num_envs} Occupancy Grids on '{self.device}':")
        print(f"  - World Bounds: {self.map_bounds}")
        print(f"  - Dimensions per grid: {self.map_dims} voxels")
        print(f"  - Voxel Size: {self.voxel_size} m")
        print(f"  - Total Voxels: {total_voxels}")
        print(f"  - Map Origin (World Coords): {self.map_origin}")

    def world_to_grid(self,  world_coords: np.ndarray, env_id) -> np.ndarray:
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
        map_origin = self.world_map_origins[env_id]
        relative_coords = world_coords - map_origin
        grid_indices = (relative_coords / self.voxel_size).astype(int)
        return grid_indices.squeeze()

    def grid_to_world(self, grid_indices: np.ndarray, env_id: int, center: bool = True) -> np.ndarray:
        if grid_indices.ndim == 1:
            grid_indices = grid_indices.reshape(1, -1)
        world_origin = self.world_map_origins[env_id]
        world_coords = (grid_indices * self.voxel_size) + world_origin
        if center:
            world_coords += (self.voxel_size / 2.0)
        return world_coords.squeeze()

    def update_occupancy(self, sensor_origins: np.ndarray, point_clouds: list[np.ndarray]):
        """
        Updates the occupancy grid with a new point cloud measurement.

        Args:
            sensor_origins (np.ndarray): A (N, 3) array for the sensor's positions.
            point_cloud (np.ndarray): An (N, 3) array of point cloud data.
        """
        valid_indices = [i for i, pc in enumerate(point_clouds) if pc.shape[0] > 0]

        if not valid_indices:
            return
        concatenated_pc = np.vstack([point_clouds[i] for i in valid_indices])
        env_indices_list = [np.full(point_clouds[i].shape[0], i, dtype=np.int32) for i in valid_indices]
        env_indices = np.concatenate(env_indices_list)
        num_total_points = concatenated_pc.shape[0]

        if num_total_points == 0:
            return
        wp_point_cloud = wp.array(concatenated_pc, dtype=wp.vec3, device=self.device)
        wp_sensor_origins = wp.array(sensor_origins, dtype=wp.vec3, device=self.device)
        wp_map_origins = wp.array(self.world_map_origins, dtype=wp.vec3, device=self.device)
        wp_env_indices = wp.array(env_indices, dtype=int, device=self.device)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=update_occupancy_fast,
            dim=num_total_points,
            inputs=[
                wp_point_cloud,
                wp_sensor_origins,
                wp_map_origins,
                wp_env_indices,
                self.voxel_size,
                wp_map_dims,
                self.occupancy_map,
                self.log_odds_free,
                self.log_odds_occupied,
            ],
            device=self.device
        )
            
        wp.launch(
            kernel=clamp_map_values,
            dim=self.occupancy_map.size,
            inputs=[self.occupancy_map, self.clamp_min, self.clamp_max],
            device=self.device
        )
        
        wp.synchronize()

    def update_visibility(self, sensor_origins: np.ndarray, point_clouds: list[np.ndarray]):
        valid_indices = [i for i, pc in enumerate(point_clouds) if pc.shape[0] > 0]
        if not valid_indices:
            return
        
        concatenated_pc = np.vstack([point_clouds[i] for i in valid_indices])
        env_indices_list = [np.full(point_clouds[i].shape[0], i, dtype=np.int32) for i in valid_indices]
        env_indices = np.concatenate(env_indices_list)
        num_total_points = concatenated_pc.shape[0]
        if num_total_points == 0:
            return        
        wp_point_cloud = wp.array(concatenated_pc, dtype=wp.vec3, device=self.device)
        wp_sensor_origins = wp.array(sensor_origins, dtype=wp.vec3, device=self.device)
        wp_map_origins = wp.array(self.world_map_origins, dtype=wp.vec3, device=self.device)
        wp_env_indices = wp.array(env_indices, dtype=int, device=self.device)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=mark_visible_voxels,
            dim=num_total_points,
            inputs=[
                wp_point_cloud,
                wp_sensor_origins,
                wp_map_origins,
                wp_env_indices,
                self.voxel_size,
                wp_map_dims,
                self.visibility_map, # Pass the visibility map here
                self.log_odds_visible, # Pass the value to add
                self.visibility_surface_hits_only 
            ],
            device=self.device
        )
        
        # Optionally clamp the visibility map to prevent values from growing infinitely
        wp.launch(
            kernel=clamp_map_values,
            dim=self.visibility_map.size,
            inputs=[self.visibility_map, 0.0, 1.0], # Min is 0
            device=self.device
        )
    def update_visitation(self, robot_positions: np.ndarray):
        num_envs = robot_positions.shape[0]
        if num_envs == 0:
            return
        wp_robot_positions = wp.array(robot_positions, dtype=wp.vec3, device=self.device)
        wp_world_map_origins = wp.array(self.world_map_origins, dtype=wp.vec3, device=self.device)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=update_visitation_kernel,
            dim=num_envs, # Launch one thread per environment
            inputs=[
                self.visitation_map,
                wp_robot_positions,
                wp_world_map_origins,
                self.voxel_size,
                wp_map_dims,
                self.num_voxels_per_map,
            ],
            device=self.device,
        )


    def reset_map(self, env_ids: list[int] = None):
        if not env_ids:
            return  
            
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return
        
        env_ids_to_reset_wp = wp.array(env_ids, dtype=int, device=self.device)
        
        # Total number of voxels to reset across all specified environments
        total_voxels_to_reset = num_envs_to_reset * self.num_voxels_per_map
        
        wp.launch(
            kernel=reset_maps_kernel,
            dim=total_voxels_to_reset, # Launch one thread for each voxel
            inputs=[
                self.occupancy_map,
                self.visibility_map,
                self.visitation_map,
                env_ids_to_reset_wp,
                self.num_voxels_per_map,
            ],
            device=self.device
        )
        # wp.launch(
        #     kernel=reset_maps_kernel,
        #     dim=total_voxels_to_reset,
        #     inputs=[self.visibility_map,
        #              env_ids_to_reset_wp, 
        #              self.num_voxels_per_map],
        #     device=self.device
        # )

        wp.synchronize()

    def get_occupied_voxels(self, env_id: int, map_origin: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Retrieves the world coordinates of occupied voxels.

        Args:
            threshold (float): The probability threshold to consider a voxel occupied.
                               Log-odds are converted to probability for this check.

        Returns:
            np.ndarray: An (M, 3) array of world coordinates for the centers of occupied voxels.
        """
        log_odds_threshold = np.log(threshold / (1.0 - threshold))
        
        # Copy map from GPU to CPU
        map_data_cpu = self.occupancy_map.numpy()
        start_index = env_id * self.num_voxels_per_map
        end_index = start_index + self.num_voxels_per_map
        env_map_data = map_data_cpu[start_index:end_index]
        
        # Find indices of occupied voxels
        occupied_indices_linear = np.where(env_map_data > log_odds_threshold)[0]

        
        if occupied_indices_linear.size == 0:
            return np.empty((0, 3))
            
        # Convert linear indices to 3D grid indices
        z_indices = occupied_indices_linear % self.map_dims[2]
        y_indices = (occupied_indices_linear // self.map_dims[2]) % self.map_dims[1]
        x_indices = occupied_indices_linear // (self.map_dims[1] * self.map_dims[2])
        
        grid_indices = np.vstack((x_indices, y_indices, z_indices)).T
        
        # Convert grid indices to world coordinates (voxel centers)
        occupied_centers = self.grid_to_world(
                                            grid_indices,
                                            env_id,
                                            center=True)
        return occupied_centers

    def get_voxel_states_as_points(self, env_id: int):
        """
        Retrieves the states of all voxels for a given environment as points and colors.
        - Occupied: Black
        - Free: White
        - Unknown: Grey
        Note: Displaying all 'unknown' voxels can be computationally intensive.
              For large maps, consider visualizing only 'occupied' and 'free' states.
        """
        map_size = self.num_voxels_per_map
        map_offset = env_id * map_size

        env_map_np = self.occupancy_map.numpy()[map_offset : map_offset + map_size]

        # Define masks for each state
        occupied_mask = env_map_np > self.log_odds_occupied
        free_mask = env_map_np < self.log_odds_free
        unknown_mask = (~occupied_mask & ~free_mask)

        # Get linear indices for each state
        occupied_indices = np.where(occupied_mask)[0]
        free_indices = np.where(free_mask)[0]
        unknown_indices = np.where(unknown_mask)[0] # Only show non-neutral unknowns

        # all_indices = np.concatenate([occupied_indices, free_indices, unknown_indices])
        all_indices = occupied_indices

        if all_indices.size == 0:
            return np.array([]), np.array([])

        # Assign colors based on state
        colors = np.zeros((len(all_indices), 3))
        colors[:len(occupied_indices)] = [0.0, 0.0, 0.0]      # Black
        colors[len(occupied_indices):len(occupied_indices) + len(free_indices)] = [1.0, 1.0, 1.0] # White
        colors[len(occupied_indices) + len(free_indices):] = [0.5, 0.5, 0.5] # Grey

        # Convert linear indices to 3D grid coordinates
        z = all_indices % self.map_dims[2]
        y = (all_indices // self.map_dims[2]) % self.map_dims[1]
        x = all_indices // (self.map_dims[1] * self.map_dims[2])

        grid_indices = np.vstack([x, y, z]).T
        world_points = self.grid_to_world(
                                        grid_indices,
                                        env_id,
                                        center=False) # Use voxel corners for viz

        return world_points, colors

    def get_local_maps(self, robot_positions_w: np.ndarray):
        
        num_req_envs = robot_positions_w.shape[0]
        if num_req_envs != self.num_envs:
            raise ValueError(f"Provided robot_positions_w has {num_req_envs} envs, but mapper is configured for {self.num_envs}.")

        local_dims = (21, 21, 11)
        local_map_dims_wp = wp.vec3i(local_dims[0], local_dims[1], local_dims[2])
        num_voxels_per_local_map = local_dims[0] * local_dims[1] * local_dims[2]
        total_local_voxels = num_req_envs * num_voxels_per_local_map

        # Prepare inputs for the kernel
        wp_robot_pos = wp.array(robot_positions_w, dtype=wp.vec3, device=self.device)
        wp_world_map_origins = wp.array(self.world_map_origins, dtype=wp.vec3, device=self.device)
        wp_global_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        # Create output arrays on the GPU
        wp_local_occ_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)
        wp_local_visibility_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)
        wp_local_visit_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)

        # Launch the extraction kernel
        wp.launch(
            kernel=extract_local_maps_kernel,
            dim=total_local_voxels,
            inputs=[
                # Global Maps
                self.occupancy_map,
                self.visibility_map,
                self.visitation_map,

                #Local Maps
                wp_local_occ_map,
                wp_local_visibility_map,
                wp_local_visit_map,

                wp_robot_pos,
                wp_world_map_origins,
                self.voxel_size,
                wp_global_map_dims,
                local_map_dims_wp,
                self.num_voxels_per_map,
                self.log_odds_neutral, # Value for out-of-bounds occupancy
            ],
            device=self.device
        )
        wp.synchronize()

        # Convert to PyTorch tensors and reshape
        local_occ_torch = wp.to_torch(wp_local_occ_map).view(num_req_envs, *local_dims)
        local_vis_torch = wp.to_torch(wp_local_visibility_map).view(num_req_envs, *local_dims)
        local_visit_torch = wp.to_torch(wp_local_visit_map).view(num_req_envs, *local_dims)

        return local_occ_torch, local_vis_torch, local_visit_torch
    
    def update_visualization(self, robot_pos: np.ndarray, robot_quat: np.ndarray):
        """Updates the interactive visualization for the configured environment."""
        # This method is now called internally by update()
        if self.visualizer is None:
            return

        points, colors = self.get_voxel_states_as_points(self.vis_env_id)
        self.visualizer.update_grid(points, colors)

        self.visualizer.update_robot_pose(robot_pos, robot_quat)