import numpy as np
import warp as wp
import open3d as o3d
from .warp_occupancy_fast import update_occupancy_fast, clamp_map_values

class OccupancyGridMapper:
    """
    A class to manage and update a 3D occupancy grid using GPU-accelerated
    fast voxel traversal with NVIDIA Warp.
    """
    def __init__(
                self,
                num_envs,
                map_dims=(128, 128, 64),
                voxel_size=0.1, 
                map_origin=(-6.4, -6.4, 0.0),
                device="cuda"):
        """
        Initializes the occupancy grid mapper.

        Args:
            map_dims (tuple): The dimensions of the grid in voxels (X, Y, Z).
            voxel_size (float): The size of each voxel in meters.
            map_origin (tuple): The world coordinate (X, Y, Z) of the grid's corner.
            device (str): The compute device for Warp ("cuda" or "cpu").
        """
        self.num_envs = num_envs
        self.device = device
        self.map_dims = np.array(map_dims, dtype=np.int32)
        self.voxel_size = float(voxel_size)
        self.map_origin = np.array(map_origin, dtype=np.float32)
        
        # Log-odds parameters
        self.log_odds_free = -0.4  # P(free) = 0.4
        self.log_odds_occupied = 0.85 # P(occupied) = 0.7
        self.clamp_min = -5.0      # Min log-odds
        self.clamp_max = 5.0       # Max log-odds
        
        # Initialize the occupancy map on the GPU
        # A value of 0.0 represents a 50% probability (unknown state).
        num_voxels = int(np.prod(self.map_dims))
        self.occupancy_map = wp.zeros(self.num_envs, num_voxels, dtype=float, device=self.device)
        print(f"Initialized Occupancy Grid on '{self.device}':")
        print(f"  - Dimensions: {self.map_dims} voxels")
        print(f"  - Voxel Size: {self.voxel_size} m")
        print(f"  - Map Origin: {self.map_origin} m")

    def update(self, sensor_origins: np.ndarray, point_clouds: np.ndarray):
        """
        Updates the occupancy grid with a new point cloud measurement.

        Args:
            sensor_origins (np.ndarray): A (N, 3) array for the sensor's positions.
            point_cloud (np.ndarray): An (N, 3) array of point cloud data.
        """
        if point_clouds.shape[1] == 0:
            return
        wp_map_dims = wp.ivec3(self.map_dims[0], self.map_dims[1], self.map_dims[2])
        # Convert inputs to Warp arrays
        for i in range(self.num_envs):
            num_points = point_clouds[i].shape[0]
            if num_points == 0:
                continue
            sensor_origin = sensor_origins[i].cpu().numpy()
            wp_sensor_origin = wp.vec3(sensor_origin[0], sensor_origin[1], sensor_origin[2])
            wp_point_cloud = wp.array(point_clouds[i], dtype=wp.vec3, device=self.device)
            wp_map_dims = wp.ivec3(self.map_dims[0], self.map_dims[1], self.map_dims[2])
            
            # Launch the kernels
            wp.launch(
                kernel=update_occupancy_fast,
                dim=num_points,
                inputs=[
                    wp_point_cloud,
                    wp_sensor_origin,
                    wp.vec3(self.map_origin),
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

    def get_occupied_voxels(self, threshold: float = 0.7) -> np.ndarray:
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
        
        # Find indices of occupied voxels
        occupied_indices_linear = np.where(map_data_cpu > log_odds_threshold)[0]
        
        if occupied_indices_linear.size == 0:
            return np.empty((0, 3))
            
        # Convert linear indices to 3D grid indices
        z_indices = occupied_indices_linear % self.map_dims[2]
        y_indices = (occupied_indices_linear // self.map_dims[2]) % self.map_dims[1]
        x_indices = occupied_indices_linear // (self.map_dims[1] * self.map_dims[2])
        
        grid_indices = np.vstack((x_indices, y_indices, z_indices)).T
        
        # Convert grid indices to world coordinates (voxel centers)
        occupied_centers = (grid_indices * self.voxel_size) + self.map_origin + (self.voxel_size / 2.0)
        return occupied_centers

    def visualize(self, point_cloud: np.ndarray, sensor_origin: np.ndarray):
        """
        Visualizes the current occupancy grid, the point cloud, and sensor origin.
        """
        occupied_centers = self.get_occupied_voxels()
        
        # Create an Open3D VoxelGrid from the occupied centers
        voxel_grid = o3d.geometry.VoxelGrid.create_from_points(
            o3d.utility.Vector3dVector(occupied_centers),
            voxel_size=self.voxel_size
        )
        voxel_grid.paint_uniform_color([0.0, 0.5, 1.0]) # Blue color for voxels

        # Create a point cloud geometry for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red for point cloud

        # Create a sphere for the sensor origin
        sensor_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=self.voxel_size * 2)
        sensor_mesh.translate(sensor_origin)
        sensor_mesh.paint_uniform_color([0.0, 1.0, 0.0]) # Green for sensor origin

        print(f"Visualizing {len(voxel_grid.get_voxels())} occupied voxels...")
        o3d.visualization.draw_geometries([voxel_grid, pcd, sensor_mesh])