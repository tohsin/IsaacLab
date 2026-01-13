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
    surface_hits_only: bool
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

    if surface_hits_only:
        end_pos_relative = ray_end - map_origin
        end_X = int(wp.floor(end_pos_relative[0] / voxel_size))
        end_Y = int(wp.floor(end_pos_relative[1] / voxel_size))
        end_Z = int(wp.floor(end_pos_relative[2] / voxel_size))

        if (end_X >= 0 and end_X < map_dims[0] and
                end_Y >= 0 and end_Y < map_dims[1] and
                end_Z >= 0 and end_Z < map_dims[2]):
            linear_index_end =  map_offset + \
                                end_X * map_dims[1] * map_dims[2] + \
                                end_Y * map_dims[2] + end_Z
            wp.atomic_add(visibility_map, linear_index_end, mark_visible_value)
    else:
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

        # wp.atomic_add(occupancy_map, linear_index_end, log_odds_occupied)