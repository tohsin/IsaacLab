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
    # Side-facing camera for the inspection task.
    inspection_camera: CameraCfg = CameraCfg(
        # prim_path="/World/envs/env_.*/Robot/base_link/inspection_camera",
        prim_path="/World/envs/env_.*/Robot/jackal_basic/tilt_link/inspection_camera",
        update_period=0.1,
        height=_height,
        width=_width,
        data_types=["rgb", "semantic_segmentation", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            #pos=(0.1, 0.0, 0.0), 
            # rot=(0, 0, -0.7071068, 0.7071068),
            rot=(0.7071068, 0, 0, -0.7071068),
            convention="ros"
        ),
        colorize_semantic_segmentation=False,
        semantic_filter=f'class:{env_parameters["semantics_name"]}',
        update_latest_camera_pose=True,
        debug_vis=True
    )
    
    # Ray-caster for accurately identifying mesh faces.
    face_raycaster: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.1,
        data_types=["face_ids"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            rot=(0, 0, -0.7071068, 0.7071068), 
            convention="ros"),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            height=_height,
            width=_width, 
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        mesh_prim_paths=[env_parameters["inspection_goal_prim_path"]],
        update_mesh_ids=True,
        debug_vis=False
    )