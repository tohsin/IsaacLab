import numpy as np
import warp as wp
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from isaaclab.utils.math import convert_quat

from .kernels.reset_kernel import reset_maps_kernel
from .kernels.occ_map_kernel import update_occupancy_fast
from .kernels.surface_visiliblity_kernel import mark_visible_voxels
from .kernels.warp_occupancy_fast import (
    update_visitation_kernel,
    extract_local_maps_kernel,
    clamp_map_values, 
)
from .run_config import map_channels, map_view_mode, visualisation_mode


class SpatialVisualizer:
    """Manages an interactive Open3D visualization window."""
    def __init__(self, voxel_size: float, title="Voxel Grid", update_frequency: int = 1):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title)

        self.voxel_size = voxel_size
        self.grid = o3d.geometry.VoxelGrid()
        self._initialized = False
        self.robot_pose_marker = None
        
        self.update_frequency = update_frequency
        self.frame_count = 0

    def update_grid(self, points, colors):
        """Updates the point cloud geometry in the visualizer."""
        self.frame_count += 1
        
        # Always pump events to keep the UI responsive for zooming/panning
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Throttle the actual geometry rebuild to save CPU
        if self.frame_count % self.update_frequency != 0:
            return
            
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
            view_ctl.set_zoom(0.6)  # Increased from 0.1 to zoom out the camera

        else:
            self.vis.remove_geometry(self.grid, reset_bounding_box=False)
            self.grid = new_grid # Combine the new grid into our existing one
            self.vis.add_geometry(self.grid, reset_bounding_box=False)
    
    def update_robot_pose(self, position: np.ndarray, orientation_quat: np.ndarray):
        """Creates or updates a coordinate frame representing the robot's pose."""
        if self.frame_count % self.update_frequency != 0:
            return
            
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

class SpatialStateManager:
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
                local_map_dims : tuple = (21, 21, 11),
                egocentric_map : bool = True,
                log_odds_free : float = -0.4,
                log_odds_occupied : float = 0.8,
                clamp_min : float = -5.0,
                clamp_max : float = 5.0,
                visualize_env_id: int | None = None,
                visualization_mode: any = None,
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
        self.local_map_dims = local_map_dims
        self.egocentric_map = egocentric_map
        self.visibility_surface_hits_only = visibility_surface_hits_only
        self.visualization_mode = visualization_mode

        self.map_origin = np.array([
            self.map_bounds['x_min'],
            self.map_bounds['y_min'],
            self.map_bounds['z_min']
        ])
        self.world_map_origins = self.env_origins + self.map_origin
         # Call this after setting map_bounds and resolution
        self.initialize_map_dimensions()
        
        # Log-odds parameters
        self.log_odds_free = log_odds_free  # P(free) = 0.4
        self.log_odds_occupied = log_odds_occupied # P(occupied) = 0.7
        self.log_odds_neutral = 0.0
        self.clamp_min = clamp_min      # Min log-odds
        self.clamp_max = clamp_max       # Max log-odds

        self.log_odds_visible = 0.4
        
        self.visualizer = None
        self.vis_env_id = visualize_env_id
        if self.vis_env_id is not None:
            if self.vis_env_id >= self.num_envs:
                print(f"Warning: visualize_env_id {self.vis_env_id} is out of bounds. Disabling visualization.")
                self.vis_env_id = None
            else:
                self.visualizer = SpatialVisualizer(voxel_size=self.voxel_size, title=f"Voxel Grid (Env {self.vis_env_id})")
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

        # Pre-allocate world map origins to avoid Host-To-Device copies during simulation
        self.wp_world_map_origins = wp.array(self.world_map_origins, dtype=wp.vec3, device=self.device)

        # Pre-allocate local maps to avoid massive memory fragmentation and allocations every step
        total_local_voxels = self.num_envs * self.local_map_dims[0] * self.local_map_dims[1] * self.local_map_dims[2]
        self.wp_local_occ_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)
        self.wp_local_visibility_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)
        self.wp_local_visit_map = wp.zeros(total_local_voxels, dtype=float, device=self.device)



        print(f"Initialized {self.num_envs} Occupancy Grids on '{self.device}':")
        print(f"  - World Bounds: {self.map_bounds}")
        print(f"  - Dimensions per grid: {self.map_dims} voxels")
        print(f"  - Voxel Size: {self.voxel_size} m")
        print(f"  - Total Voxels: {total_voxels}")
        print(f"  - Map Origin (World Coords): {self.map_origin}")
        print(f"  - Local Map Frame: {'egocentric' if self.egocentric_map else 'allocentric'}")

    def world_to_grid(self,  world_coords: np.ndarray, env_id) -> np.ndarray:
        is_1d = world_coords.ndim == 1
        if is_1d:
            world_coords = world_coords.reshape(1, -1)
        map_origin = self.world_map_origins[env_id]
        relative_coords = world_coords - map_origin
        grid_indices = (relative_coords / self.voxel_size).astype(int)
        return grid_indices[0] if is_1d else grid_indices

    def grid_to_world(self, grid_indices: np.ndarray, env_id: int, center: bool = True) -> np.ndarray:
        is_1d = grid_indices.ndim == 1
        if is_1d:
            grid_indices = grid_indices.reshape(1, -1)
        world_origin = self.world_map_origins[env_id]
        world_coords = (grid_indices * self.voxel_size) + world_origin
        if center:
            world_coords += (self.voxel_size / 2.0)
        return world_coords[0] if is_1d else world_coords

    def update_occupancy(self, sensor_origins: torch.Tensor, point_clouds: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]):
        """
        Updates the occupancy grid with a new point cloud measurement.

        Args:
            sensor_origins (torch.Tensor): A (N, 3) tensor for the sensor's positions.
            point_clouds (list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]): A list of point cloud data or a flattened tuple.
        """
        if isinstance(point_clouds, tuple):
            concatenated_pc, env_indices = point_clouds
            num_total_points = concatenated_pc.shape[0]
        else:
            valid_indices = [i for i, pc in enumerate(point_clouds) if pc.shape[0] > 0]
            if not valid_indices:
                return
            concatenated_pc = torch.cat([point_clouds[i] for i in valid_indices], dim=0)
            env_indices_list = [torch.full((point_clouds[i].shape[0],), i, dtype=torch.int32, device=self.device) for i in valid_indices]
            env_indices = torch.cat(env_indices_list, dim=0)
            num_total_points = concatenated_pc.shape[0]

        if num_total_points == 0:
            return
        wp_point_cloud = wp.from_torch(concatenated_pc.contiguous(), dtype=wp.vec3)
        wp_sensor_origins = wp.from_torch(sensor_origins.contiguous(), dtype=wp.vec3)
        wp_env_indices = wp.from_torch(env_indices.contiguous(), dtype=wp.int32)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=update_occupancy_fast,
            dim=num_total_points,
            inputs=[
                wp_point_cloud,
                wp_sensor_origins,
                self.wp_world_map_origins,
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

    def update_visibility(self, sensor_origins: torch.Tensor, point_clouds: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]):
        if isinstance(point_clouds, tuple):
            concatenated_pc, env_indices = point_clouds
            num_total_points = concatenated_pc.shape[0]
        else:
            valid_indices = [i for i, pc in enumerate(point_clouds) if pc.shape[0] > 0]
            if not valid_indices:
                return
            concatenated_pc = torch.cat([point_clouds[i] for i in valid_indices], dim=0)
            env_indices_list = [torch.full((point_clouds[i].shape[0],), i, dtype=torch.int32, device=self.device) for i in valid_indices]
            env_indices = torch.cat(env_indices_list, dim=0)
            num_total_points = concatenated_pc.shape[0]
        
        if num_total_points == 0:
            return        
        wp_point_cloud = wp.from_torch(concatenated_pc.contiguous(), dtype=wp.vec3)
        wp_sensor_origins = wp.from_torch(sensor_origins.contiguous(), dtype=wp.vec3)
        wp_env_indices = wp.from_torch(env_indices.contiguous(), dtype=wp.int32)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=mark_visible_voxels,
            dim=num_total_points,
            inputs=[
                wp_point_cloud,
                wp_sensor_origins,
                self.wp_world_map_origins,
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
        
        wp.synchronize()
    
    def update_visitation(self, robot_positions: torch.Tensor):
        num_envs = robot_positions.shape[0]
        if num_envs == 0:
            return
        wp_robot_positions = wp.from_torch(robot_positions.contiguous(), dtype=wp.vec3)
        wp_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

        wp.launch(
            kernel=update_visitation_kernel,
            dim=num_envs, # Launch one thread per environment
            inputs=[
                self.visitation_map,
                wp_robot_positions,
                self.wp_world_map_origins,
                self.voxel_size,
                wp_map_dims,
                self.num_voxels_per_map,
            ],
            device=self.device,
        )
        
        wp.synchronize()

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

    def get_collision_bounds(self, shift_x=None, shift_y=None, shift_z=None):
        """
        Returns the bounding box (min_x, max_x, min_y, max_y, min_z, max_z) in local map indices
        for the collision check zone.
        """
        local_dims = self.local_map_dims
        center_x = local_dims[0] // 2
        center_y = local_dims[1] // 2
        center_z = 0 # Robot is at the floor level in the local map
        
        target_radius_m = 0.6 # Increased to 0.6m (3 voxels) to enforce a larger personal space
        calculated_shift = max(1, int(np.ceil(target_radius_m / self.resolution)))
        
        sx = shift_x if shift_x is not None else calculated_shift
        sy = shift_y if shift_y is not None else calculated_shift
        sz = shift_z if shift_z is not None else calculated_shift

        min_x = max(0, center_x - sx)
        max_x = min(local_dims[0], center_x + sx + 1)
        min_y = max(0, center_y - sy)
        max_y = min(local_dims[1], center_y + sy + 1)
        min_z = max(1, center_z) # Skip the floor level (z=0)
        max_z = min(local_dims[2], center_z + sz + 1)

        return min_x, max_x, min_y, max_y, min_z, max_z

    def get_voxel_states_as_points(self, env_id: int, robot_pos_w: torch.Tensor = None, robot_quat_w: torch.Tensor = None):
        """
        Retrieves the states of voxels for visualization based on the map mode and channel.
        """
        all_indices = np.array([], dtype=int)
        colors = np.array([])
        world_points = np.array([])

        if self.visualization_mode.map_mode == map_view_mode.LOCAL:
            if robot_pos_w is None or robot_quat_w is None:
                return np.array([]), np.array([])
            
            # Use yaw-only rotation for local map extraction if egocentric
            from isaaclab.utils.math import yaw_quat
            if getattr(self, "egocentric_map", True):
                robot_yaw_quat_w = yaw_quat(robot_quat_w)
            else:
                robot_yaw_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(robot_pos_w.shape[0], 1)
            
            local_occ, local_vis, local_visit = self.get_local_maps(robot_pos_w, robot_yaw_quat_w)
            
            if self.visualization_mode.channel == map_channels.OCCUPANCY:
                local_map_np = local_occ[env_id].cpu().numpy()
                mask = local_map_np > 0.0 # Standard threshold for log-odds occupancy (P > 0.5)
            elif self.visualization_mode.channel == map_channels.VISIBILITY:
                local_map_np = local_vis[env_id].cpu().numpy()
                mask = local_map_np >= 0.4
            elif self.visualization_mode.channel == map_channels.VISITATION:
                local_map_np = local_visit[env_id].cpu().numpy()
                mask = local_map_np > 0
            elif self.visualization_mode.channel == map_channels.COLLISION:
                local_map_np = local_occ[env_id].cpu().numpy()
                mask = local_map_np > 0.0 # Standard threshold for log-odds occupancy
                
                # Add bounding box mask
                min_x, max_x, min_y, max_y, min_z, max_z = self.get_collision_bounds()
                
                boundary_mask = np.zeros_like(local_map_np, dtype=bool)
                # Wireframe edges for the bounding box
                mx = max_x - 1 if max_x > 0 else 0
                my = max_y - 1 if max_y > 0 else 0
                mz = max_z - 1 if max_z > 0 else 0
                
                # Z edges
                boundary_mask[min_x, min_y, min_z:max_z] = True
                boundary_mask[min_x, my, min_z:max_z] = True
                boundary_mask[mx, min_y, min_z:max_z] = True
                boundary_mask[mx, my, min_z:max_z] = True

                # Y edges
                boundary_mask[min_x, min_y:max_y, min_z] = True
                boundary_mask[min_x, min_y:max_y, mz] = True
                boundary_mask[mx, min_y:max_y, min_z] = True
                boundary_mask[mx, min_y:max_y, mz] = True

                # X edges
                boundary_mask[min_x:max_x, min_y, min_z] = True
                boundary_mask[min_x:max_x, min_y, mz] = True
                boundary_mask[min_x:max_x, my, min_z] = True
                boundary_mask[min_x:max_x, my, mz] = True
                
                mask = mask | boundary_mask
            else:
                return np.array([]), np.array([])

            if np.any(mask):
                x, y, z = np.where(mask)
                colors = np.zeros((len(x), 3))
                if self.visualization_mode.channel == map_channels.OCCUPANCY:
                    colors[:] = [0.0, 0.0, 0.0]
                elif self.visualization_mode.channel == map_channels.VISIBILITY:
                    intensities = local_map_np[x, y, z]
                    colors[:, 1] = intensities
                    colors[:, 2] = 0.2
                elif self.visualization_mode.channel == map_channels.VISITATION:
                    counts = local_map_np[x, y, z]
                    visitation_threshold = 10.0  # Adjust this to tune how fast it turns red
                    norm_c = np.clip(counts / visitation_threshold, 0, 1)
                    colors[:, 0] = norm_c # Red
                    colors[:, 2] = 1.0 - norm_c # Blue
                elif self.visualization_mode.channel == map_channels.COLLISION:
                    min_x, max_x, min_y, max_y, min_z, max_z = self.get_collision_bounds()
                    
                    for i in range(len(x)):
                        in_box = (min_x <= x[i] < max_x) and \
                                 (min_y <= y[i] < max_y) and \
                                 (min_z <= z[i] < max_z)
                        is_occupied = local_map_np[x[i], y[i], z[i]] > 1.1
                        
                        if in_box and is_occupied:
                            colors[i] = [1.0, 0.0, 0.0] # Red for collision
                        elif local_map_np[x[i], y[i], z[i]] > 0.0:
                            # colors[i] = [0.8, 0.8, 0.8] # Light grey for normal occupancy
                            colors[i] = [1.0, 1.0, 1.0] # White for normal occupancy
                        else:
                            colors[i] = [0.0, 0.0, 1.0] # Blue for boundary box

                # Convert indices to local coordinates
                # Z starts at 0 (the floor), so we do not shift Z by half the dimension
                center = np.array([self.local_map_dims[0] // 2, self.local_map_dims[1] // 2, 0])
                grid_indices = np.vstack([x, y, z]).T
                # Add 0.5 to pass the *center* of the voxel to Open3D to prevent floating point aliasing
                world_points = (grid_indices - center + 0.5) * self.voxel_size
            
            return world_points, colors

        elif self.visualization_mode.map_mode == map_view_mode.GLOBAL:
            map_size = self.num_voxels_per_map
            map_offset = env_id * map_size

            if self.visualization_mode.channel == map_channels.OCCUPANCY:
                env_map_np = self.occupancy_map.numpy()[map_offset : map_offset + map_size]
                occupied_mask = env_map_np > 0.0 # Standard threshold for log-odds occupancy (P > 0.5)
                all_indices = np.where(occupied_mask)[0]

                if all_indices.size > 0:
                    colors = np.zeros((len(all_indices), 3))
                    colors[:] = [0.0, 0.0, 0.0]

            elif self.visualization_mode.channel == map_channels.VISIBILITY:
                env_map_np = self.visibility_map.numpy()[map_offset : map_offset + map_size]
                visible_mask = env_map_np >= 0.4
                all_indices = np.where(visible_mask)[0]

                if all_indices.size > 0:
                    intensities = env_map_np[all_indices]
                    colors = np.zeros((len(all_indices), 3))
                    colors[:, 1] = intensities
                    colors[:, 2] = 0.2
            
            elif self.visualization_mode.channel == map_channels.VISITATION:
                env_map_np = self.visitation_map.numpy()[map_offset : map_offset + map_size]
                visited_mask = env_map_np > 0
                all_indices = np.where(visited_mask)[0]
                
                if all_indices.size > 0:
                    counts = env_map_np[all_indices]
                    visitation_threshold = 10.0  # Adjust this to tune how fast it turns red
                    norm_c = np.clip(counts / visitation_threshold, 0, 1)
                    colors = np.zeros((len(all_indices), 3))
                    colors[:, 0] = norm_c
                    colors[:, 2] = 1.0 - norm_c

            if all_indices.size == 0:
                return np.array([]), np.array([])

            # Convert linear indices to 3D grid coordinates
            z = all_indices % self.map_dims[2]
            y = (all_indices // self.map_dims[2]) % self.map_dims[1]
            x = all_indices // (self.map_dims[1] * self.map_dims[2])

            grid_indices = np.vstack([x, y, z]).T
            world_points = self.grid_to_world(
                                            grid_indices,
                                            env_id,
                                            center=True) # Use voxel centers for viz to prevent floating point aliasing

            return world_points, colors
        
        return np.array([]), np.array([])

    def get_local_maps(self, robot_positions_w: torch.Tensor, robot_quats_w: torch.Tensor):
        """Extract robot-centered maps using Isaac Lab ``wxyz`` quaternions."""
        
        num_req_envs = robot_positions_w.shape[0]
        if num_req_envs != self.num_envs:
            raise ValueError(f"Provided robot_positions_w has {num_req_envs} envs, but mapper is configured for {self.num_envs}.")

        local_dims = self.local_map_dims
        local_map_dims_wp = wp.vec3i(local_dims[0], local_dims[1], local_dims[2])
        num_voxels_per_local_map = local_dims[0] * local_dims[1] * local_dims[2]
        total_local_voxels = num_req_envs * num_voxels_per_local_map

        # Prepare inputs for the kernel
        wp_robot_pos = wp.from_torch(robot_positions_w.contiguous(), dtype=wp.vec3)
        # Isaac Lab stores quaternions as (w, x, y, z), whereas Warp's
        # ``wp.quat`` memory layout is (x, y, z, w). Convert at this boundary
        # so both egocentric yaw rotations and the allocentric identity are
        # interpreted correctly by ``wp.quat_rotate`` in the extraction kernel.
        robot_quats_xyzw = convert_quat(robot_quats_w, to="xyzw").contiguous()
        wp_robot_quat = wp.from_torch(robot_quats_xyzw, dtype=wp.quat)
        wp_global_map_dims = wp.vec3i(self.map_dims[0], self.map_dims[1], self.map_dims[2])

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
                self.wp_local_occ_map,
                self.wp_local_visibility_map,
                self.wp_local_visit_map,

                wp_robot_pos,
                wp_robot_quat,
                self.wp_world_map_origins,
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
        local_occ_torch = wp.to_torch(self.wp_local_occ_map).view(num_req_envs, *local_dims)
        local_vis_torch = wp.to_torch(self.wp_local_visibility_map).view(num_req_envs, *local_dims)
        local_visit_torch = wp.to_torch(self.wp_local_visit_map).view(num_req_envs, *local_dims)

        return local_occ_torch, local_vis_torch, local_visit_torch
    
    def update_visualization(self, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor):
        """Updates the interactive visualization for the configured environment."""
        # This method is now called internally by update()
        if self.visualizer is None or self.visualization_mode is None:
            return

        points, colors = self.get_voxel_states_as_points(self.vis_env_id, robot_pos_w, robot_quat_w)
        self.visualizer.update_grid(points, colors)

        if self.visualization_mode.map_mode == map_view_mode.LOCAL:
            # For local egocentric view, keep robot at the origin
            self.visualizer.update_robot_pose(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        else:
            # For global view, move the robot pose marker to its actual location
            self.visualizer.update_robot_pose(robot_pos_w[self.vis_env_id].cpu().numpy(), robot_quat_w[self.vis_env_id].cpu().numpy())
