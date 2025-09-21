import warp as wp
from typing import Any
wp.init()

@wp.kernel
def update_occupancy_fast(
    point_cloud: wp.array(dtype=wp.vec3),
    sensor_origin: wp.vec3,
    map_origin: wp.vec3,
    voxel_size: float,
    map_dims: wp.ivec3,
    occupancy_map: wp.array(dtype=float), # Using float for log-odds
    log_odds_free: float,      # A small negative value, e.g., -0.4
    log_odds_occupied: float,  # A positive value, e.g., 0.8
):
    tid = wp.tid()


    ray_start = sensor_origin
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

    stepX = 1 if ray_dir[0] >= 0 else -1
    stepY = 1 if ray_dir[1] >= 0 else -1
    stepZ = 1 if ray_dir[2] >= 0 else -1

    epsilon = 1e-6
    safe_ray_dir_x = ray_dir[0] if wp.abs(ray_dir[0]) > epsilon else epsilon * stepX
    safe_ray_dir_y = ray_dir[1] if wp.abs(ray_dir[1]) > epsilon else epsilon * stepY
    safe_ray_dir_z = ray_dir[2] if wp.abs(ray_dir[2]) > epsilon else epsilon * stepZ
# Distance to cross the first voxel boundary
    next_voxel_boundary_x = float(X + (stepX > 0)) * voxel_size
    next_voxel_boundary_y = float(Y + (stepY > 0)) * voxel_size
    next_voxel_boundary_z = float(Z + (stepZ > 0)) * voxel_size
    
    tMaxX = (next_voxel_boundary_x - start_pos_relative[0]) / safe_ray_dir_x
    tMaxY = (next_voxel_boundary_y - start_pos_relative[1]) / safe_ray_dir_y
    tMaxZ = (next_voxel_boundary_z - start_pos_relative[2]) / safe_ray_dir_z

    # Distance to travel one voxel width along the ray
    tDeltaX = voxel_size / wp.abs(safe_ray_dir_x)
    tDeltaY = voxel_size / wp.abs(safe_ray_dir_y)
    tDeltaZ = voxel_size / wp.abs(safe_ray_dir_z)

    # --- 3. Incremental Traversal (Main Loop) ---
    t_current = 0.0
    while t_current < ray_dist:
        # Update current voxel as 'free'
        if (X >= 0 and X < map_dims[0] and
            Y >= 0 and Y < map_dims[1] and
            Z >= 0 and Z < map_dims[2]):
            linear_index = X * map_dims[1] * map_dims[2] + Y * map_dims[2] + Z
            wp.atomic_add(occupancy_map, linear_index, log_odds_free)

        # Advance to the next voxel
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                t_current = tMaxX
                X += stepX
                tMaxX += tDeltaX
            else:
                t_current = tMaxZ
                Z += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                t_current = tMaxY
                Y += stepY
                tMaxY += tDeltaY
            else:
                t_current = tMaxZ
                Z += stepZ
                tMaxZ += tDeltaZ

    # --- 4. Mark Endpoint as Occupied ---
    end_pos_relative = ray_end - map_origin
    end_X = int(wp.floor(end_pos_relative[0] / voxel_size))
    end_Y = int(wp.floor(end_pos_relative[1] / voxel_size))
    end_Z = int(wp.floor(end_pos_relative[2] / voxel_size))

    if (end_X >= 0 and end_X < map_dims[0] and
        end_Y >= 0 and end_Y < map_dims[1] and
        end_Z >= 0 and end_Z < map_dims[2]):
        linear_index_end = end_X * map_dims[1] * map_dims[2] + end_Y * map_dims[2] + end_Z
        wp.atomic_add(occupancy_map, linear_index_end, log_odds_occupied)

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