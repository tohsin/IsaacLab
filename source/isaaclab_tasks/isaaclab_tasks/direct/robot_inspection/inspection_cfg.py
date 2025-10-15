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
from .configs.config_ import ROBOT_CONFIGS, Env_params
env_parameters = Env_params["empty_room"]


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
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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


    sphere_cfg = RigidObjectCfg( 
        prim_path="/World/envs/env_.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 5.0, 0.05))
    )
    cone_cfg = RigidObjectCfg( 
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.3,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 15.0, 0.05))
    )
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
    reward_cfg : RewardsCfg = RewardsCfg()
    sensor_cfg : SensorsCfg = SensorsCfg()
    robot_phys_cfg: RobotPhysicsCfg = RobotPhysicsCfg()
    mapping_cfg: MappingCfg = MappingCfg()
    
    action_dim = NotImplementedError
    if isinstance(action_space, spaces.Discrete):
        action_dim = action_space.n 
    else:
        action_dim = action_space.shape[0]

    observation_space = spaces.Dict({

        "robot-pose": spaces.Box(
            low=float("-inf"), 
            high=float("inf"),
            shape=(13 + action_dim,),
            dtype=np.float32
        ),
        "cameras": spaces.Box(
            low=float("-inf"),
            high=float("inf"), 
            shape=(_height, _height, 6)
        ),
        "local-map": spaces.Box(
            low=float("-inf"),
            high=float("inf"), 
            # Shape is (X, Y, Z, Channels). We have 2 channels: Occupancy, Visibility, Visitation
            shape=(21, 21, 11, 3), 
            dtype=np.float32,
        )
    })

    # scene
    scene: WarehouseSceneCfg = WarehouseSceneCfg(
        num_envs=16, 
        env_spacing= 60.0, 
        replicate_physics=True
    )

    max_robot_distance = 2000

   
 
    # inspection
    init_inspection_threshold = 0.5# Coverage % threshold to count as valid inspection
    init_spatial_level = 4
    max_inspection_threshold = 0.99
    curriculum_difficulty_increment = 0.05


    terminate_on_all_inspected = True
    min_episode_length = 2000
    max_faces_to_inspect = env_parameters["num_faces"]

    # Logging
    logging_interval: int = 1000

