import warp as wp
from typing import Any
wp.init()


@wp.kernel
def update_visitation_kernel(
    visitation_map: wp.array(dtype=float),
    robot_positions: wp.array(dtype=wp.vec3),
    map_origins: wp.array(dtype=wp.vec3),
    voxel_size: float,
    map_dims: wp.vec3i,
    num_voxels_per_map: int,
):
    env_id = wp.tid() # in this case each parale process is an env
    robot_pos = robot_positions[env_id]
    map_origin = map_origins[env_id]
    relative_pos = robot_pos - map_origin
    ix = int(wp.floor(relative_pos[0] / voxel_size))
    iy = int(wp.floor(relative_pos[1] / voxel_size))
    iz = int(wp.floor(relative_pos[2] / voxel_size))

    if ix >= 0 and ix < map_dims[0] and iy >= 0 and iy < map_dims[1] and iz >= 0 and iz < map_dims[2]:
        # Calculate linear index
        linear_index =  ix * map_dims[1] * map_dims[2] + iy * map_dims[2] + iz
        env_offset = env_id * num_voxels_per_map
        
        # Atomically increment the visitation count
        wp.atomic_add(visitation_map, env_offset + linear_index, 1.0)

@wp.kernel
def extract_local_maps_kernel(
    # Input global maps
    global_occupancy_map: wp.array(dtype=float),
    global_visibility_map: wp.array(dtype=float),
    global_visitation_map: wp.array(dtype=float), 
    # Output local maps (flattened)
    local_occupancy_map: wp.array(dtype=float),
    local_visibility_map: wp.array(dtype=float),
    local_visitation_map: wp.array(dtype=float),
    # Robot and map info
    robot_positions_w: wp.array(dtype=wp.vec3),
    map_origins: wp.array(dtype=wp.vec3),
    voxel_size: float,
    global_map_dims: wp.vec3i,
    local_map_dims: wp.vec3i,
    num_voxels_per_global_map: int,
    out_of_bounds_value: float,
):
    """
    Extracts a local map centered around each robot from the global maps.
    Each thread corresponds to one voxel in one of the local maps.
    """
    tid = wp.tid()

    # 1. Deconstruct thread ID to find (env_id, local_x, local_y, local_z)
    num_voxels_per_local_map = local_map_dims[0] * local_map_dims[1] * local_map_dims[2]
    env_id = tid // num_voxels_per_local_map
    map_origin_for_env = map_origins[env_id]
    local_linear_index = tid % num_voxels_per_local_map

    lz = local_linear_index % local_map_dims[2]
    ly = (local_linear_index // local_map_dims[2]) % local_map_dims[1]
    lx = local_linear_index // (local_map_dims[1] * local_map_dims[2])

    # 2. Find the corresponding global grid index for this local voxel
    robot_pos_w = robot_positions_w[env_id]
    
    # Convert robot's world position to its global grid index
    robot_relative_pos = robot_pos_w - map_origin_for_env
    robot_gx = int(wp.floor(robot_relative_pos[0] / voxel_size))
    robot_gy = int(wp.floor(robot_relative_pos[1] / voxel_size))
    robot_gz = int(wp.floor(robot_relative_pos[2] / voxel_size))
    
    # Center the local map on the robot's XY, and use Z as the floor
    center_offset_x = local_map_dims[0] // 2
    center_offset_y = local_map_dims[1] // 2
    
    target_gx = robot_gx + lx - center_offset_x
    target_gy = robot_gy + ly - center_offset_y
    target_gz = robot_gz + lz # Voxel (lx, ly, 0) corresponds to robot's Z level

    # 3. Check if the target global index is within bounds
    if (target_gx >= 0 and target_gx < global_map_dims[0] and
        target_gy >= 0 and target_gy < global_map_dims[1] and
        target_gz >= 0 and target_gz < global_map_dims[2]):
        
        # If in bounds, read from the global map
        global_map_offset = env_id * num_voxels_per_global_map
        global_linear_index = global_map_offset + \
            target_gx * global_map_dims[1] * global_map_dims[2] + \
            target_gy * global_map_dims[2] + \
            target_gz
        
        occ_val = global_occupancy_map[global_linear_index]
        vis_val = global_visibility_map[global_linear_index]
        visit_val = global_visitation_map[global_linear_index]

        local_occupancy_map[tid] = occ_val
        local_visibility_map[tid] = vis_val
        local_visitation_map[tid] = visit_val
    else:
        # If out of bounds, write a default value
        local_occupancy_map[tid] = out_of_bounds_value
        local_visibility_map[tid] = 0.0 # OOB for visibility is just "not seen"
        local_visitation_map[tid] = 0.0
        
@wp.kernel
def clamp_map_values(
    occupancy_map: wp.array(dtype=float),
    clamp_min: float,
    clamp_max: float,
):
    """Clamps the log-odds values in the map to a specified range."""
    tid = wp.tid()
    val = occupancy_map[tid]
    if val < clamp_min:
        occupancy_map[tid] = clamp_min
    elif val > clamp_max:
        occupancy_map[tid] = clamp_max

