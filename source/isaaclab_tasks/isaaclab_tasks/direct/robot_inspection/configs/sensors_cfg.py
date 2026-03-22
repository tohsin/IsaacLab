import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .config_ import env_parameters
from isaaclab.sensors import TiledCameraCfg, RayCasterCameraCfg, patterns, MultiMeshRayCasterCameraCfg
from ..run_config import cfg_mode, record_Cfg
@configclass
class SensorsCfg:
    """Configuration for all robot-mounted sensors."""
    if cfg_mode == record_Cfg:
        camera_height: int = 512
        camera_width: int = 512
    else:
        camera_height: int = 128
        camera_width: int = 128

    nav_data_types = ["distance_to_image_plane"]
    if getattr(cfg_mode, "nav_camera_modality", "rgb") in ["rgb", "rgbd"]:
        nav_data_types.append("rgb")

    # Front-facing camera for navigation.
    navigation_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/jackal_basic/base_link/nav_camera",
        update_period=0.24,
        height=camera_height,
        width=camera_width,
        data_types=nav_data_types,
        update_latest_camera_pose=True,
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
        debug_vis=cfg_mode.debug
    )
    
    ptz_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link/ptz_camera",
        update_period=0.24,
        height=camera_height,
        width=camera_width,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation", "motion_vectors"],
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
        semantic_filter=[f'class:{name}' for name in env_parameters.semantics_name] if isinstance(env_parameters.semantics_name, list) else f'class:{env_parameters.semantics_name}',
        update_latest_camera_pose=True,
        debug_vis=cfg_mode.debug
    )

    if getattr(cfg_mode, "add_high_res_inspection_camera", False):
        high_res_ptz_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link/high_res_ptz_camera",
            update_period=0.24,
            height=getattr(cfg_mode, "high_res_camera_height", 1024),
            width=getattr(cfg_mode, "high_res_camera_width", 1024),
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
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
            semantic_filter=[f'class:{name}' for name in env_parameters.semantics_name] if isinstance(env_parameters.semantics_name, list) else f'class:{env_parameters.semantics_name}',
            update_latest_camera_pose=True,
            debug_vis=cfg_mode.debug
        )

    #Ray-caster for accurately identifying mesh faces.
    face_raycaster: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        # prim_path="/World/envs/env_.*/Robot/base_link",
        prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link",
        update_period=0.24,
        data_types=["face_ids", "normals", "distance_to_image_plane"],
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
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                target_prim_expr=target.prim_path,
                track_mesh_transforms=True
            ) for target in env_parameters.inspection_targets.values()
        ],
        update_mesh_ids=True,
        debug_vis=False
    )