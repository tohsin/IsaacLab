from isaaclab.utils import configclass
@configclass
class MappingCfg:
    """Configuration for the occupancy mapping module."""
    use_occupancy_map: bool = True
    egocentric_map: bool = False # Flag to toggle between egocentric (rotating) and allocentric (fixed) local map
    visibility_surface_hits_only: bool = True
    compute_global_map_entropy = True
    filter_floor_occupancy: bool = True # Crucial: Must be true so obstacles don't get starved out during downsampling!
    # Map update tuning
    log_odds_free: float = -0.4  # How aggressively to clear free space
    log_odds_occupied: float = 0.8 # How aggressively to mark obstacles
    clamp_min: float = -5.0 # Lower bound clipping
    clamp_max: float = 5.0  # Upper bound clipping
    resolution: float = 0.25  # Voxel size in meters
    bounds: dict = {
        "x_min": -10.5, "x_max": 9.5,
        "y_min": -12.5, "y_max": 18.0,
        "z_min": 0.0, "z_max": 2.5
    }
    map_update_interval: float = 4 # steps between map updates
    # Increased local_map_dims for testing the field of view
    local_map_dims: tuple =(21, 21, 11) # Size of the egocentric local map