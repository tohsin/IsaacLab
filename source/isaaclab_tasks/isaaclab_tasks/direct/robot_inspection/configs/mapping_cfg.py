from isaaclab.utils import configclass
@configclass
class MappingCfg:
    """Configuration for the occupancy mapping module."""
    use_occupancy_map: bool = True
    egocentric_map: bool = True  # True: robot-heading-aligned; False: world-axis-aligned local map
    visibility_surface_hits_only: bool = True
    compute_global_map_entropy = True
    filter_floor_occupancy: bool = True # Crucial: Must be true so obstacles don't get starved out during downsampling!
    # Map update tuning
    log_odds_free: float = -0.2  # How aggressively to clear free space
    log_odds_occupied: float = 0.8 # How aggressively to mark obstacles
    clamp_min: float = -5.0 # Lower bound clipping
    clamp_max: float = 5.0  # Upper bound clipping

    bounds: dict = {
        "x_min": -10.5, "x_max": 9.5,
        "y_min": -12.5, "y_max": 18.0,
        "z_min": 0.0, "z_max": 2.5
    }
    # resolution: float = 0.2 # Voxel size in meters
    # map_update_interval: float = 2 # steps between map updates
    # # Increased local_map_dims for testing the field of view
    # local_map_dims: tuple =(21, 21, 11) # Size of the egocentric local map


    resolution: float = 0.1 # Voxel size in meters
    map_update_interval: int = 1 # steps between map updates
    # Bounds peak temporary VRAM while reducing the full global maps for
    # information-gain, visibility, and episode-summary metrics.
    global_map_reward_chunk_size: int = 65536
    # Increased local_map_dims for testing the field of view
    local_map_dims: tuple =(29, 29, 15) # Size of the egocentric local map
