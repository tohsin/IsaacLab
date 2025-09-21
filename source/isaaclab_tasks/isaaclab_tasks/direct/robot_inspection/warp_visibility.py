import warp as wp
from typing import Any
wp.init()

@wp.kernel
def update_visibility_map(
    point_cloud: wp.array(dtype=wp.vec3),
    sensor_origin: wp.vec3,
    map_origin: wp.vec3,
    voxel_size: float,
    map_dims: wp.ivec3,
    visibility_map: wp.array(dtype=int),  # Integer array to store view counts
):
    """
    A Warp kernel to update a 3D visibility map.

    This kernel performs ray casting from the sensor origin to each point in the point cloud.
    It increments a counter for each voxel that falls along these rays, effectively
    counting how many times each voxel has been observed.

    Args:
        point_cloud (wp.array): An array of 3D points representing the current sensor reading.
        sensor_origin (wp.vec3): The 3D position of the sensor in the world frame.
        map_origin (wp.vec3): The 3D position of the corner of the map's bounding box.
        voxel_size (float): The side length of a single voxel.
        map_dims (wp.ivec3): The dimensions of the voxel grid (width, height, depth).
        visibility_map (wp.array): A flattened 1D array of integers representing the 3D visibility map.
                                    The value in each cell is the number of times it has been observed.
    """
    tid = wp.tid()

    ray_start = sensor_origin
    ray_end = point_cloud[tid]
    ray_vec = ray_end - ray_start
    ray_dist = wp.length(ray_vec)

    if ray_dist < 1e-6:
        return

    ray_dir = ray_vec / ray_dist
    step_size = voxel_size * 0.5  # Step along the ray at half-voxel increments
    dist_traveled = 0.0

    while dist_traveled < ray_dist:
        current_pos = ray_start + ray_dir * dist_traveled
        voxel_idx = wp.ivec3((current_pos - map_origin) / voxel_size)

        # Check if the voxel is within the map boundaries
        if (voxel_idx[0] >= 0 and voxel_idx[0] < map_dims[0] and
            voxel_idx[1] >= 0 and voxel_idx[1] < map_dims[1] and
            voxel_idx[2] >= 0 and voxel_idx[2] < map_dims[2]):

            # Convert 3D voxel index to a 1D linear index
            linear_index = voxel_idx[0] * map_dims[1] * map_dims[2] + \
                           voxel_idx[1] * map_dims[2] + \
                           voxel_idx[2]

            # Atomically increment the visibility count for this voxel
            wp.atomic_add(visibility_map, linear_index, 1)

        dist_traveled += step_size