from isaaclab.utils import configclass
@configclass
class MappingCfg:
    """Configuration for the occupancy mapping module."""
    use_occupancy_map: bool = True
    visibility_surface_hits_only: bool = True
    compute_global_map_entropy = True
    resolution: float = 0.1  # Voxel size in meters
    bounds: dict = {
        "x_min": -10.5, "x_max": 9.5,
        "y_min": -12.5, "y_max": 18.0,
        "z_min": 0.0, "z_max": 2.5
    }
    map_update_interval: float = 4 # steps between map updates