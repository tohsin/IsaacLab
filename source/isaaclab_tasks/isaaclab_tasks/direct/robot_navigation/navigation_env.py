# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
from __future__ import annotations

from collections import deque
import gymnasium as gym
import torch
from collections.abc import Sequence
import numpy as np

from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.semantics import add_labels

from isaaclab.sensors import Camera
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
import isaacsim.core.utils.stage as stage_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
# from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul, quat_apply, quat_conjugate, quat_apply_inverse
# from semanc_manager import SemanticManager, add_semantic_tags_from_config
from .navigation_cfg import Isaac3dNavigationEnvCfg
import wandb

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import time 
import torch.nn.functional as F
import warp as wp
visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
debug = False
use_wandb =  True #not debug

class Isaac3dNavigationEnv(DirectRLEnv):
    cfg: Isaac3dNavigationEnvCfg

    def __init__(self, cfg: Isaac3dNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Multi env navigation env")

        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale


        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w

         
        self._setup_tensor_buffers()

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        self.last_log_step = 0
        self.visualization_timer = 0
        self.visualization_interval = 10


    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
        self.init_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_quats = torch.zeros((self.num_envs, 4), device=self.device)

        action_shape = (self.num_envs, self.cfg.action_space.shape[0])

        self.last_action = torch.zeros(action_shape, device=self.device)

        self.prev_dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.success_rate = 0.0
        # Buffers for exploration rewards
        self.cached_rewards = {}
    

    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.goal_marker = RigidObject(self.cfg.scene.goal_marker)
        self.scene.rigid_objects["goal_marker"] = self.goal_marker

        self._obs_camera = Camera(self.cfg.sensor_cfg.navigation_camera)
        self.scene.sensors["camera"] = self._obs_camera

        self.cone = RigidObject(self.cfg.cone_cfg)
        self.scene.rigid_objects["cone"] = self.cone

        self.sphere = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere

        self.scene.clone_environments(copy_from_source=False)
        # clone and replicate
    
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.last_action = self.actions.clone()
    
    def _apply_action(self) -> None:
        
        linear_cmd = self.actions[:, 0] * self.cfg.robot_phys_cfg.max_linear_velocity  # Forward/Backward command
        ang_cmd = self.actions[:, 1] * self.cfg.robot_phys_cfg.max_angular_velocity  # Left/Right turn command

        w_sep = self.cfg.robot_phys_cfg.wheel_separation
        w_rad = self.cfg.robot_phys_cfg.wheel_radius
        
        left_vel = (linear_cmd - (ang_cmd * w_sep / 2)) / w_rad
        right_vel = (linear_cmd + (ang_cmd * w_sep / 2)) / w_rad

        
        # Clamp wheel velocities to avoid exceeding max limits
        max_wheel_v = self.cfg.robot_phys_cfg.max_wheel_velocity
        left_vel = torch.clamp(left_vel, -max_wheel_v, max_wheel_v)
        right_vel = torch.clamp(right_vel, -max_wheel_v, max_wheel_v)

        
        self.wheel_commands = torch.stack([left_vel, right_vel,
                                    left_vel, right_vel], dim=1)
        
        # Scale the wheel commands
        target = self.wheel_commands * self.cfg.action_scale
            # x = target

        # print(f"[INFO] Wheel Commands: {self.wheel_commands.clone()}")
        self.robot.set_joint_velocity_target(target, joint_ids=self._wheel_joint_indices)
    
    def _compute_goal_observations(self, robot_pos):
        target_vec = self.goal_pos[:, :2] - robot_pos[:, :2]
        return torch.norm(target_vec, dim=1)
    
    def _get_observations(self) -> dict:
        pose_world = self.robot.data.root_state_w.clone()

        position = pose_world[...,    :3]
        quat = pose_world[..., 3:7]
        lin_vel = pose_world[..., 7:10]
        ang_vel = pose_world[..., 10:13]
        dist = self._compute_goal_observations(position)
        self.dist_to_goal = dist.clone()
        cam_data = self._obs_camera.data.output[ "rgb"] / 255.0
        # # Use pure rgb or depth information

        lin_vel_body = quat_apply_inverse(quat, lin_vel)
        ang_vel_body = quat_apply_inverse(quat, ang_vel)
        full_obs = torch.cat([
            position, quat, lin_vel, ang_vel, self.last_action, dist
        ], dim=-1)

        obs =   {
            'robot-pose': full_obs,
            'cameras': cam_data.clone(),
        }
        return {"policy": obs}

    # def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
    #     if self.common_step_counter % self.cfg.logging_interval == 0:
            
    #         log_data = {
              
    #             # Add step-wise rewards (from the cache)
    #         }
    #         if self.cached_rewards:
    #             log_data.update(self.cached_rewards)
    #         if use_wandb:
    #             wandb.log(log_data, step=self.common_step_counter)
    #     return super().step(action)

    def _get_rewards(self) -> torch.Tensor:
        """
            Face Coverage Rewards,
            Exploration Rewards,
            Visibility Rewards
        """
        progress = (self.prev_dist_to_goal - self.dist_to_goal)
        r_progress = progress * self.cfg.reward_cfg.progress_reward_scale

        is_reached = self.dist_to_goal < self.cfg.goal_dist_threshold
        r_success = is_reached.float() * self.cfg.reward_cfg.goal_reached_bonus
        r_penalty = self.cfg.reward_cfg.time_penalty
        total_reward = r_progress + r_success + r_penalty

        self.prev_dist_to_goal = self.dist_to_goal.clone()
        self.success_buf = is_reached.long()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for timeout

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        goal_reached = self.dist_to_goal < self.cfg.goal_dist_threshold

        return goal_reached, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES


        super()._reset_idx(env_ids)
            #number of steps taken in the episode
         
        
        # Sample random positions within specified range
        num_resets = len(env_ids)
        new_vel = torch.zeros((num_resets, 3), device=self.device)
        new_pos, new_quat = torch.zeros((num_resets, 3), device=self.device)

        fixed_goal = torch.tensor([3.0, 3.0, 0.0], device=self.device)
        goal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)

        self.goal_marker.write_root_pose_to_sim(
            torch.cat([self.goal_pos[env_ids], goal_quat], dim=-1), 
            env_ids
        )
        # Combine into root state
        new_root_state = torch.cat([new_pos, new_quat, new_vel, torch.zeros((num_resets, 3), device=self.device)], dim=-1)
        
        # Add environment origins
        new_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Reset joint positions and velocities to default
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        # Write states to simulation
        self.robot.write_root_pose_to_sim(new_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(new_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        root_state = self.robot.data.root_state_w[env_ids]
        self.init_position[env_ids] = root_state[:, :3]
        self.init_quats[env_ids] = root_state[:, 3:7]

        current_robot_pos = self.robot.data.root_pos_w[env_ids]
        start_dist = self._compute_goal_observations(current_robot_pos)
        self.dist_to_goal[env_ids] = start_dist
        self.prev_dist_to_goal[env_ids] = start_dist