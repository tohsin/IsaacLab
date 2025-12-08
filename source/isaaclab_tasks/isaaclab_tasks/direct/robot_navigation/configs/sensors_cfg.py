import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .config_ import env_parameters
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, patterns, MultiMeshRayCasterCameraCfg

@configclass
class SensorsCfg:
    """Configuration for all robot-mounted sensors."""
    _width: int = 128
    _height: int = 128

    # Front-facing camera for navigation.
    navigation_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
        update_period=0.1,
        height=_height,
        width=_width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.3, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros"
        ),
        debug_vis=False 
    )
  