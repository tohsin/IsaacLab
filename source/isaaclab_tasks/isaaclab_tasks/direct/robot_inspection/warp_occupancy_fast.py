import warp as wp
from typing import Any
wp.init()

@wp.kernel
def mark_visible_voxels(
    point_cloud: wp.array(dtype=wp.vec3),
    sensor_origins: wp.array(dtype=wp.vec3),
    map_origins: wp.array(dtype=wp.vec3),
    env_indices: wp.array(dtype=int),
    # Map parameters
    voxel_size: float,
    map_dims: wp.vec3i,
    visibility_map: wp.array(dtype=float), # Using float for log-odds
    # Log-odds values
    mark_visible_value: float,
):
    tid = wp.tid()
    # indexing by environment
    env_id = env_indices[tid]
    ray_start = sensor_origins[env_id]
    map_origin = map_origins[env_id]


    ray_end = point_cloud[tid]
    ray_vec = ray_end - ray_start
    ray_dist = wp.length(ray_vec)

    if ray_dist < 1e-6:
        return
    
    ray_dir = ray_vec / ray_dist


    start_pos_relative = ray_start - map_origin
    X = int(wp.floor(start_pos_relative[0] / voxel_size))
    Y = int(wp.floor(start_pos_relative[1] / voxel_size))
    Z = int(wp.floor(start_pos_relative[2] / voxel_size))

    stepX = wp.where(ray_dir[0] >= 0.0, 1.0, -1.0)
    stepY = wp.where(ray_dir[1] >= 0.0, 1.0, -1.0)
    stepZ = wp.where(ray_dir[2] >= 0.0, 1.0, -1.0)
    epsilon = 1e-6
    safe_ray_dir_x = wp.where(wp.abs(ray_dir[0]) > epsilon, ray_dir[0], epsilon * stepX)
    safe_ray_dir_y = wp.where(wp.abs(ray_dir[1]) > epsilon, ray_dir[1], epsilon * stepY)
    safe_ray_dir_z = wp.where(wp.abs(ray_dir[2]) > epsilon, ray_dir[2], epsilon * stepZ)

    
    # Distance to cross the first voxel boundary
    next_voxel_boundary_x = float(X + wp.where(stepX > 0.0, 1, 0)) * voxel_size
    next_voxel_boundary_y = float(Y + wp.where(stepY > 0.0, 1, 0)) * voxel_size
    next_voxel_boundary_z = float(Z + wp.where(stepZ > 0.0, 1, 0)) * voxel_size
    
    tMaxX = (next_voxel_boundary_x - start_pos_relative[0]) / safe_ray_dir_x
    tMaxY = (next_voxel_boundary_y - start_pos_relative[1]) / safe_ray_dir_y
    tMaxZ = (next_voxel_boundary_z - start_pos_relative[2]) / safe_ray_dir_z

    # Distance to travel one voxel width along the ray
    tDeltaX = voxel_size / wp.abs(safe_ray_dir_x)
    tDeltaY = voxel_size / wp.abs(safe_ray_dir_y)
    tDeltaZ = voxel_size / wp.abs(safe_ray_dir_z)

    # --- 3. Incremental Traversal (Main Loop) ---
    t_current = float(0.0)
    num_voxels_per_map = map_dims[0] * map_dims[1] * map_dims[2]
    map_offset = env_id * num_voxels_per_map

    while t_current < ray_dist:
        # Update current voxel as 'free'
        if (X >= 0 and X < map_dims[0] and
            Y >= 0 and Y < map_dims[1] and
            Z >= 0 and Z < map_dims[2]):
            linear_index = map_offset + \
                X * map_dims[1] * map_dims[2] +\
                Y * map_dims[2] +\
                Z
            wp.atomic_add(visibility_map, linear_index, mark_visible_value)

        # Advance to the next voxel
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                t_current = tMaxX
                X += int(stepX)
                tMaxX += tDeltaX
            else:
                t_current = tMaxZ
                Z += int(stepZ)
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                t_current = tMaxY
                Y += int(stepY)
                tMaxY += tDeltaY
            else:
                t_current = tMaxZ
                Z += int(stepZ)
                tMaxZ += tDeltaZ

@wp.kernel
def update_occupancy_fast(
    point_cloud: wp.array(dtype=wp.vec3),
    sensor_origins: wp.array(dtype=wp.vec3),
    map_origins: wp.array(dtype=wp.vec3),
    env_indices: wp.array(dtype=int),
    # Map parameters
    voxel_size: float,
    map_dims: wp.vec3i,
    occupancy_map: wp.array(dtype=float),  # Using float for log-odds
    # Log-odds values
    mark_visible: float, # A small negative value, e.g., -0.4
):
    tid = wp.tid()
    # indexing by environment
    env_id = env_indices[tid]
    ray_start = sensor_origins[env_id]
    map_origin = map_origins[env_id]


    ray_end = point_cloud[tid]
    ray_vec = ray_end - ray_start
    ray_dist = wp.length(ray_vec)

    if ray_dist < 1e-6:
        return
    
    ray_dir = ray_vec / ray_dist


    start_pos_relative = ray_start - map_origin
    X = int(wp.floor(start_pos_relative[0] / voxel_size))
    Y = int(wp.floor(start_pos_relative[1] / voxel_size))
    Z = int(wp.floor(start_pos_relative[2] / voxel_size))

    stepX = wp.where(ray_dir[0] >= 0.0, 1.0, -1.0)
    stepY = wp.where(ray_dir[1] >= 0.0, 1.0, -1.0)
    stepZ = wp.where(ray_dir[2] >= 0.0, 1.0, -1.0)
    epsilon = 1e-6
    safe_ray_dir_x = wp.where(wp.abs(ray_dir[0]) > epsilon, ray_dir[0], epsilon * stepX)
    safe_ray_dir_y = wp.where(wp.abs(ray_dir[1]) > epsilon, ray_dir[1], epsilon * stepY)
    safe_ray_dir_z = wp.where(wp.abs(ray_dir[2]) > epsilon, ray_dir[2], epsilon * stepZ)

    
    # Distance to cross the first voxel boundary
    next_voxel_boundary_x = float(X + wp.where(stepX > 0.0, 1, 0)) * voxel_size
    next_voxel_boundary_y = float(Y + wp.where(stepY > 0.0, 1, 0)) * voxel_size
    next_voxel_boundary_z = float(Z + wp.where(stepZ > 0.0, 1, 0)) * voxel_size
    
    tMaxX = (next_voxel_boundary_x - start_pos_relative[0]) / safe_ray_dir_x
    tMaxY = (next_voxel_boundary_y - start_pos_relative[1]) / safe_ray_dir_y
    tMaxZ = (next_voxel_boundary_z - start_pos_relative[2]) / safe_ray_dir_z

    # Distance to travel one voxel width along the ray
    tDeltaX = voxel_size / wp.abs(safe_ray_dir_x)
    tDeltaY = voxel_size / wp.abs(safe_ray_dir_y)
    tDeltaZ = voxel_size / wp.abs(safe_ray_dir_z)

    # --- 3. Incremental Traversal (Main Loop) ---
    t_current = float(0.0)
    num_voxels_per_map = map_dims[0] * map_dims[1] * map_dims[2]
    map_offset = env_id * num_voxels_per_map

    while t_current < ray_dist:
        # Update current voxel as 'free'
        if (X >= 0 and X < map_dims[0] and
            Y >= 0 and Y < map_dims[1] and
            Z >= 0 and Z < map_dims[2]):
            linear_index = map_offset + \
                X * map_dims[1] * map_dims[2] +\
                Y * map_dims[2] +\
                Z
            wp.atomic_add(occupancy_map, linear_index, log_odds_free)

        # Advance to the next voxel
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                t_current = tMaxX
                X += int(stepX)
                tMaxX += tDeltaX
            else:
                t_current = tMaxZ
                Z += int(stepZ)
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                t_current = tMaxY
                Y += int(stepY)
                tMaxY += tDeltaY
            else:
                t_current = tMaxZ
                Z += int(stepZ)
                tMaxZ += tDeltaZ

    # --- 4. Mark Endpoint as Occupied ---
    end_pos_relative = ray_end - map_origin
    end_X = int(wp.floor(end_pos_relative[0] / voxel_size))
    end_Y = int(wp.floor(end_pos_relative[1] / voxel_size))
    end_Z = int(wp.floor(end_pos_relative[2] / voxel_size))

    if (end_X >= 0 and end_X < map_dims[0] and
        end_Y >= 0 and end_Y < map_dims[1] and
        end_Z >= 0 and end_Z < map_dims[2]):
        linear_index_end =  map_offset +\
                            end_X * map_dims[1] * map_dims[2] + \
                            end_Y * map_dims[2] + end_Z
        update_val = log_odds_occupied - log_odds_free
        wp.atomic_add(occupancy_map, linear_index_end, update_val)
        # wp.atomic_add(occupancy_map, linear_index_end, log_odds_occupied)

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


@wp.kernel
def reset_maps_kernel(
    occupancy_map: wp.array(dtype=float),
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
    final_voxel_index = map_start_index + voxel_index_in_map
    
    occupancy_map[final_voxel_index] = 0.0