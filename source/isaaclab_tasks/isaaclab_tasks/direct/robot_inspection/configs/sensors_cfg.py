import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .config_ import env_parameters
from isaaclab.sensors import TiledCameraCfg, RayCasterCameraCfg, patterns, MultiMeshRayCasterCameraCfg
_debug_vis = False
@configclass
class SensorsCfg:
    """Configuration for all robot-mounted sensors."""
    camera_height: int = 64
    camera_width: int = 64

    # Front-facing camera for navigation.
    navigation_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/jackal_basic/base_link/nav_camera",
        update_period=0.24,
        height=camera_height,
        width=camera_width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.3, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros"
        ),
        debug_vis=_debug_vis
    )
    ptz_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link/ptz_camera",
        update_period=0.24,
        height=camera_height,
        width=camera_width,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation" ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            rot=(0.7071068, 0, 0, -0.7071068),
            convention="ros"
        ),
        colorize_semantic_segmentation=False,
        semantic_filter=f'class:{env_parameters["semantics_name"]}',
        update_latest_camera_pose=True,
        debug_vis=_debug_vis
    )

    #Ray-caster for accurately identifying mesh faces.
    face_raycaster: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        # prim_path="/World/envs/env_.*/Robot/base_link",
        prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link",
        update_period=0.24,
        data_types=["face_ids"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            #rot=(0, 0, -0.7071068, 0.7071068),
            rot=(0.7071068, 0, 0, -0.7071068),
            convention="ros"
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            height=camera_height,
            width=camera_width, 
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        mesh_prim_paths=[env_parameters["inspection_goal_prim_path"]],
        update_mesh_ids=True,
        debug_vis=_debug_vis
    )