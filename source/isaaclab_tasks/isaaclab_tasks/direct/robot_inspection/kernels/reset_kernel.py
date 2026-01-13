import warp as wp
from typing import Any
wp.init()

@wp.kernel
def reset_maps_kernel(
    occupancy_map: wp.array(dtype=float),
    visibility_map: wp.array(dtype=float), 
    visitation_map: wp.array(dtype=float),
    env_ids_to_reset: wp.array(dtype=int),
    num_voxels_per_map: int,
):
    """Resets the occupancy map to zero."""
    tid = wp.tid()
    env_index_in_list = tid // num_voxels_per_map
    # Determine which voxel within that environment's map this thread should reset
    voxel_index_in_map = tid % num_voxels_per_map

    # Get the actual environment ID from the input list
    env_id = env_ids_to_reset[env_index_in_list]

    # Calculate the final linear index in the global occupancy_map array
    map_start_index = env_id * num_voxels_per_map
    global_voxel_index = map_start_index + voxel_index_in_map
    
    occupancy_map[global_voxel_index] = 0.0
    visibility_map[global_voxel_index] = 0.0
    visitation_map[global_voxel_index] = 0.0 