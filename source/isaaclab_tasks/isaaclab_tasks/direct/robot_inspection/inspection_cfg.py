# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np


from isaaclab.envs.utils import spaces
from isaaclab.sensors.camera import tiled_camera
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from gymnasium.spaces.discrete import Discrete
from isaaclab.sensors import TiledCameraCfg, CameraCfg, RayCasterCameraCfg, patterns, MultiMeshRayCasterCameraCfg
from gymnasium import spaces

# from semantic_manager import SemanticManager
ROBOT_CONFIGS = {
    "jackal": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal_basic.usd",
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 4  # 4 wheels
    },
    "jetbot": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd", 
        "wheel_joint_expr": ".*wheel.*",
        "action_space": 2  # 2 wheels
    },
}
Env_params = {
    "Brick":{
        "num_faces": 12_000,
        "semantics_name": "Brick",
        "file_name": "/home/tosin/Desktop/IsaacLab/environments/ware_house_semantic.usd",
        "prim_path": "/World/ground/terrain/ware_house_brick/_61_foam_brick"
    },

    'Brick_default':{
        "num_faces": 12_000,
        "semantics_type": "class",
        "semantics_name": "brick",
        "file_name": "/home/tosin/IsaacLab_inspection/environments/ware_house_brick.usd",
        "prim_path": "/World/ground/terrain/_61_foam_brick",

    },
    'complex_forklift':{
        "num_faces": 21_000,
        "semantics_type": "class",
        "semantics_name": "forklift",
        "file_name": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
        "prim_path": "/World/ground/terrain/forklift"
    },
    'complex_forklift_2':{
        "num_faces": 21_000,
        "semantics_type": "class",
        "semantics_name": "forklift",
        "env_file_path": "/home/tosin/IsaacLab_inspection/environments/small_forklift.usd",
        "inspection_goal_prim_path": "/World/envs/env_.*/warehouse/forklift",
        "env_prim_path": "/World/envs/env_.*/warehouse"
    }
}
env_parameters = Env_params["complex_forklift_2"]

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    # scene
    warehouse: AssetBaseCfg = AssetBaseCfg(
        prim_path = env_parameters["env_prim_path"],
        spawn=sim_utils.UsdFileCfg(
            usd_path=env_parameters["env_file_path"],
            # scale=(1.0, 1.0, 1.0),
        ),
        collision_group=-1, # Keep collision settings
        debug_vis=True
    )


@configclass
class Isaac3dinspectionEnvCfg(DirectRLEnvCfg):
    # env
    env_parameters = env_parameters
    use_camera_obs: bool = True
    _width, _height = 128, 128
    inspection_objective_prim_path = env_parameters["inspection_goal_prim_path"]


    decimation = 2
    # semantic_config_path = "source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/semantic_config_warehouse.json"
    episode_length_s = 42
    action_scale = 1.0  # [N]
    #action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    # action_space = spaces.Discrete(3)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    #behind the shelves
    # viewer = ViewerCfg( eye=(-24, 29, 8.4), lookat=(-13, 27.6, 0.0))
    # next to the Goal
    # viewer = ViewerCfg( eye=(-24, 5, 8.4), lookat=(0, 0, 0.0))
    # next to the Goal CLOSER
    viewer = ViewerCfg( eye=(-10, 5, 8.4), lookat=(0, 0, 0.0))
    
    # outside wall
    # viewer = ViewerCfg( eye=(-30, 5, 8.4), lookat=(0, 0, 0.0))
    state_space = 0
    wheel_velocity_scale = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 1 / 129, 
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type="tgs",
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=1
        ))

    # robot
    wheel_seperation = 	0.37558 # 0.37558
    wheel_radius = 0.098
    forward_vel = 5.5
    turn_vel = 5.0

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_CONFIGS["jackal"]["usd_path"],
        ),
        actuators={
            "wheel_acts": ImplicitActuatorCfg(
                joint_names_expr=ROBOT_CONFIGS["jackal"]["wheel_joint_expr"],
                damping=None,
                stiffness=None
            )
        },
        debug_vis=True
    
    )
    #SENSOR (Raycaster + Tiled Camera)
    #Front Camera for Navigation
    observation_camera = CameraCfg(
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
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.3, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros"
        ),
        debug_vis=False  # Disable for performance
    )
    # Side Camera for 3D Inspection
    inspection_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/inspection_camera",
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
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            rot = (0,  0, -0.7071068, 0.7071068),
            convention="ros"
        ),
        colorize_semantic_segmentation=False,
        debug_vis=False  # Disable for performance
    )
    face_Camera_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.1,
        data_types=["face_ids"],#face_ids
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),
            rot = (0,  0, -0.7071068, 0.7071068),
            convention="ros"
        ),
        pattern_cfg= patterns.PinholeCameraPatternCfg(
            height=_height,
            width=_width,
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        mesh_prim_paths = [inspection_objective_prim_path],
        update_mesh_ids=True,
        debug_vis=True
    )
    LOCAL_MAP_SIZE = 40

    observation_space = spaces.Dict({

        "robot-pose": spaces.Box(
            low=float("-inf"), 
            high=float("inf"),
            shape=(13 + action_space.shape[0],),
            dtype=np.float32
        ),
        "cameras": spaces.Box(
            low=float("-inf"),
            high=float("inf"), 
            shape=(_height, _height, 6)
        )
    })

    # scene
    scene: WarehouseSceneCfg = WarehouseSceneCfg(
        num_envs=16, 
        env_spacing= 50.0, 
        replicate_physics=True
    )

    max_robot_distance = 2000

    #reward

    mesh_coverage_reward_scale = 0.0001  # Scale for inspection coverage reward
    ent_IG_reward_scale = 7e-4  # Scale for information gain reward via entropy reduction
    visibility_IG_reward_scale = 1e-3  # Scale for visibility information gain reward
    distance_reward_scale = 1.0  # Scale for distance-based rewards
    distance_reward_scale_beta = 2.0  # Beta parameter for distance-based rewards
    max_reward_distance = 3.0 #max distance for reward
    time_penalty = -0.001
    spin_penalty_scale = 0.05
    movement_reward_scale = 0.05

    # inspection
    init_inspection_threshold = 0.3 # Coverage % threshold to count as valid inspection
    init_spatial_level = 6
    max_inspection_threshold = 0.95
    curriculum_difficulty_increment = 0.05
    coverage_reward = 3.0
    # inspection_save_dir = "inspection_captures"

    terminate_on_all_inspected = True
    min_episode_length = 2500
    max_faces_to_inspect = env_parameters["num_faces"]

    max_linear_velocity = 2.0
    max_angular_velocity = 4.0
    min_discovery_interval = 1.0
    max_wheel_velocity = 20.41  # Max wheel velocity for the robot

    # Logging
    logging_interval: int = 200