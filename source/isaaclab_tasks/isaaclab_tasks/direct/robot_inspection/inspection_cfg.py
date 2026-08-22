# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.envs.utils import spaces
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from gymnasium.spaces.discrete import Discrete
from gymnasium import spaces

from .configs.sensors_cfg import SensorsCfg
from .configs.mapping_cfg import MappingCfg
from .configs.rewards_cfg import RewardsCfg
from  .configs.robot_cfg import RobotPhysicsCfg
# from semantic_manager import SemanticManager
from .configs.config_ import ROBOT_CONFIGS, env_parameters
from .run_config import cfg_mode

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    # scene
    warehouse: AssetBaseCfg = AssetBaseCfg(
        prim_path = env_parameters.prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=env_parameters.usd_path,
            # scale=(1.0, 1.0, 1.0),
        ),
        collision_group=-1, # Keep collision settings
        debug_vis=cfg_mode.debug
    )

@configclass
class Isaac3dinspectionEnvCfg(DirectRLEnvCfg):
    # env
    env_parameters = inspection_goal_cfg = env_parameters
    use_camera_obs: bool = True


    decimation = 12 # 10.75 Hz control frequency (New)
    # decimation = 6  # 21.5 Hz control frequency (Old)
    # semantic_config_path = "source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/semantic_config_warehouse.json"
    episode_length_s = 42
    action_scale = 0.8  # [N]
    '''
        Action Space Discrete(5):
            - [v_high, ω_zero] (Go Straight Fast)
            - [v_mid, ω_zero] (Go Straight Slow)

            - [v_mid, ω_high_left] (Turn Left)
            - [v_mid, ω_high_right] (Turn Right)

            - [v_zero, ω_high_left] (Rotate in Place left)
            - [v_zero, ω_high_right] (Rotate in Place right)
    '''
    # action_space = spaces.Discrete(6)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
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


    max_obstacles: int = getattr(cfg_mode, "max_obstacles", 6)    # Cap obstacles around 10 depending on the environment scale

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_CONFIGS["jackal_ptz"]["usd_path"],
            activate_contact_sensors=True,
        ),
        actuators={
            "wheel_acts": ImplicitActuatorCfg(
                joint_names_expr=ROBOT_CONFIGS["jackal_ptz"]["wheel_joint_expr"],
                damping=10_000, #10000.0,
                stiffness=None
            ),
            "ptz_acts": ImplicitActuatorCfg(
                joint_names_expr=ROBOT_CONFIGS["jackal_ptz"]["ptz_joint_expr"], 
                stiffness=0.0, # High stiffness drives the joint to a specific angle
                damping= 200.0     # Moderate damping prevents oscillation #  40 -Position control
            )
        },
        debug_vis=cfg_mode.debug
    
    )
    reward_cfg : RewardsCfg = RewardsCfg()
    sensor_cfg : SensorsCfg = SensorsCfg()
    robot_phys_cfg: RobotPhysicsCfg = RobotPhysicsCfg()
    mapping_cfg: MappingCfg = MappingCfg()
    # Direct/debug runs take their map-frame selection from run_config.py.
    mapping_cfg.egocentric_map = getattr(cfg_mode, "egocentric_map", mapping_cfg.egocentric_map)
    
    action_dim = NotImplementedError
    if isinstance(action_space, spaces.Discrete):
        action_dim = action_space.n 
    else:
        action_dim = action_space.shape[0]

    observation_space = spaces.Dict({

        "robot-pose": spaces.Box(
            low=float("-inf"), 
            high=float("inf"),
            shape=(13 + action_dim + 2,), # Plus ptz joint position
            dtype=np.float32
        ),
        "cameras": spaces.Box(
            low=float("-inf"),
            high=float("inf"), 
            shape=(sensor_cfg.camera_width, sensor_cfg.camera_height, 8 if getattr(cfg_mode, "nav_camera_modality", "rgb") == "rgbd" else (7 if getattr(cfg_mode, "nav_camera_modality", "rgb") == "rgb" else 5))
        ),
        "local-map": spaces.Box(
            low=float("-inf"),
            high=float("inf"), 
            # Shape is (X, Y, Z, Channels). We have 2 channels: Occupancy, Visibility, Visitation
            shape=(*mapping_cfg.local_map_dims, 3), 
            dtype=np.float32,
        )
    })

    # scene
    scene: WarehouseSceneCfg = WarehouseSceneCfg(
        num_envs=16, 
        env_spacing= 35.0, 
        replicate_physics=True
    )
 
    # inspection
    init_inspection_threshold = 0.5# Coverage % threshold to count as valid inspection

    terminate_on_all_inspected = True
    min_episode_length = cfg_mode.max_episode_length

    # Logging
    logging_interval: int = cfg_mode.logging_interval
