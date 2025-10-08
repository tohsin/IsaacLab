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

from isaaclab.sensors import TiledCamera, RayCasterCamera, Camera, MultiMeshRayCasterCamera
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
import isaacsim.core.utils.stage as stage_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
# from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul, quat_apply, quat_conjugate, quat_apply_inverse
# from semanc_manager import SemanticManager, add_semantic_tags_from_config
from .inspection_cfg import Isaac3dinspectionEnvCfg
import wandb
from .currilum_manager import Curriculum
from .utils import  NormalizeReward, visualise_faces
from .occupancy_grid_mapper import OccupancyGridMapper
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import time
import torch.nn.functional as F
import warp as wp
visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
debug = False
use_wandb = not debug

class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Multi env inspection env")

        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale


        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w
         
        self._setup_tensor_buffers()

        self.curriculum = Curriculum(
            init_inspection_threshold=self.cfg.init_inspection_threshold,
            max_inspection_threshold=self.cfg.max_inspection_threshold,
            curriculum_difficulty_increment=self.cfg.curriculum_difficulty_increment,
            init_spatial_level=self.cfg.init_spatial_level,
            num_steps=30,
            num_envs=self.num_envs,
            device=self.device
        )
        if self.cfg.use_occupancy_map:
            self.occupancy_mapper = OccupancyGridMapper(
                num_envs=self.num_envs,
                map_bounds=self.cfg.occupancy_map_bounds,
                resolution=self.cfg.occupancy_map_resolution,
                device=self.device,
                # visualize_env_id= None
                visualize_env_id=0 if debug else None
            )
        self.rewardscaler = NormalizeReward(device=self.device)
        self.last_log_step = 0
        self.visualization_timer = 0
        self.visualization_interval = 10

    def close(self):
        """Cleanup for the environment."""
        if self.cfg.use_occupancy_map and self.occupancy_mapper.visualizer:
            self.occupancy_mapper.visualizer.close()
        super().close()

    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
        self.init_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_quats = torch.zeros((self.num_envs, 4), device=self.device)

        if isinstance(self.cfg.action_space, gym.spaces.Discrete):
            action_shape = (self.num_envs, 1)
        else:
            action_shape = (self.num_envs, self.cfg.action_space.shape[0])

        self.last_action = torch.zeros(action_shape, device=self.device)
        self.discovered_faces_buffer = [torch.tensor([], dtype=torch.float32, device=self.device) for _ in range(self.num_envs)]

        self.episode_log_buffer = {
            "coverage_percent": deque(maxlen=self.cfg.scene.num_envs * 4),
            "faces_discovered": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_map_entropy": deque(maxlen=self.cfg.scene.num_envs * 4), 
        }
     
        self.success_rate = 0.0
        self.last_fork_lift_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)


                # Buffers for exploration rewards
        local_map_shape = self.cfg.observation_space["local-map"].shape
        self.prev_local_occ_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_local_vis_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_coverage_ratio = torch.zeros(self.num_envs, device=self.device)

        self.cached_rewards = {}


        # Visitation mapping and marking
        self.visitation_map_resolution = 0.5
        map_bounds = self.cfg.occupancy_map_bounds
        self.visitation_map_width = int((map_bounds['x_max'] - map_bounds['x_min']) / self.visitation_map_resolution)
        self.visitation_map_height = int((map_bounds['y_max'] - map_bounds['y_min']) / self.visitation_map_resolution)
        self.visitation_map_origin = torch.tensor([map_bounds['x_min'], map_bounds['y_min']], device=self.device)

        self.visitation_maps = torch.zeros(
            (self.num_envs, self.visitation_map_width, self.visitation_map_height),
            dtype=torch.float32, # Use float to store counts for the reward calculation
            device=self.device
        )
    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot


        self._obs_camera = Camera(self.cfg.observation_camera)
        self.scene.sensors["camera"] = self._obs_camera

        self._inspection_camera = Camera(self.cfg.inspection_camera)
        self.scene.sensors["inspection_camera"] = self._inspection_camera

        #OOcclusion and Inspection Objects.
        # self.cube = RigidObject(self.cfg.cube_cfg)
        # self.scene.rigid_objects["cube"] = self.cub
        cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Rubiks_Cube/rubiks_cube.usd",
            scale=(5.0, 5.0, 5.0),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0, mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "inspection_goal")]
            )
        cfg.func(
            "/World/envs/env_.*/rubiks_cube", cfg, 
            translation=(0, -3.0, 0.2), 
            orientation=(0.70711, 0.0, 0.0, 0.70711)
        )
        
        self.cone = RigidObject(self.cfg.cone_cfg)
        self.scene.rigid_objects["cone"] = self.cone

        self.sphere = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere

        self._raycaster_camera = MultiMeshRayCasterCamera(self.cfg.face_Camera_cfg)
        self.scene.sensors["raycaster_camera"] = self._raycaster_camera


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
        
        if isinstance(self.single_action_space, gym.spaces.Box):
            linear_velocity = self.actions[:, 0] * self.cfg.max_linear_velocity  # Forward/Backward command
            angular_velocity = self.actions[:, 1] * self.cfg.max_angular_velocity  # Left/Right turn command


            left_wheel_velocity = (linear_velocity - (angular_velocity * self.cfg.wheel_separation / 2)) / self.cfg.wheel_radius
            right_wheel_velocity = (linear_velocity + (angular_velocity * self.cfg.wheel_separation / 2)) / self.cfg.wheel_radius



            # Clamp wheel velocities to avoid exceeding max limits
            left_wheel_velocity = torch.clamp(left_wheel_velocity, -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
            right_wheel_velocity = torch.clamp(right_wheel_velocity, -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)

            
            self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                        left_wheel_velocity, right_wheel_velocity], dim=1)
           
            # Scale the wheel commands
            target = self.wheel_commands * self.cfg.action_scale
            # x = target

        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            '''
                Action Space Discrete(5):
                    - [v_high, ω_zero] (Go Straight Fast)
                    - [v_mid, ω_zero] (Go Straight Slow)
                    - [v_mid, ω_high_left] (Turn Left)
                    - [v_mid, ω_high_right] (Turn Right)
                    - [v_zero, ω_high_left] (Rotate in Place)
            '''
            actions = self.actions.squeeze(-1)
            linear_velocity = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            angular_velocity = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            # Move Forward fast
            linear_velocity[actions == 0] = self.cfg.max_linear_velocity
            angular_velocity[actions == 0] = 0.0

            # Move Forward slow
            linear_velocity[actions == 1] = self.cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 1] = 0.0

            # Action 2: Turn Left , while moving forward
            linear_velocity[actions == 2] = self.cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 2] = self.cfg.max_angular_velocity

            # Action 3: Turn Right
            linear_velocity[actions == 3] = self.cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 3] = -self.cfg.max_angular_velocity

            # Action 4: Rotate in Place (Left)
            linear_velocity[actions == 4] = 0.0
            angular_velocity[actions == 4] = self.cfg.max_angular_velocity

             # Action 4: Rotate in Place (Right)
            linear_velocity[actions == 5] = 0.0
            angular_velocity[actions == 5] = -self.cfg.max_angular_velocity

            left_wheel_velocity = (linear_velocity - angular_velocity * self.cfg.wheel_separation / 2.0) / self.cfg.wheel_radius
            right_wheel_velocity = (linear_velocity + angular_velocity * self.cfg.wheel_separation / 2.0) / self.cfg.wheel_radius

            self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                       left_wheel_velocity, right_wheel_velocity], dim=1)
            target = self.wheel_commands.clone()
            target = target.view(self.num_envs, -1)

        # print(f"[INFO] Wheel Commands: {self.wheel_commands.clone()}")
        self.robot.set_joint_velocity_target(target, joint_ids=self._wheel_joint_indices)
    
    def _update_maps(self, visualise: bool = False):
        
        if not self.cfg.use_occupancy_map:
            return
        robot_pos_w = self.robot.data.root_pos_w
        robot_quat_w = self.robot.data.root_quat_w # (w, x, y, z)

        camera_local_pos = torch.tensor(self.cfg.observation_camera.offset.pos, device=self.device)
        camera_local_quat = torch.tensor(self.cfg.observation_camera.offset.rot, device=self.device)

        ## Convert ROS (x, y, z, w) to math (w, x, y, z) for quat_mul
        camera_local_quat = camera_local_quat[[3, 0, 1, 2]]

        rotated_offsets = quat_apply(robot_quat_w, camera_local_pos.expand_as(robot_pos_w))
        camera_world_pos = robot_pos_w + rotated_offsets
        camera_world_quat = quat_mul(robot_quat_w, camera_local_quat.expand_as(robot_quat_w))

        point_clouds_list  = []
        depth_data = self._obs_camera.data.output["distance_to_image_plane"]
        intrinsic_matrices = self._obs_camera.data.intrinsic_matrices
        for i in range(self.num_envs):

            pointcloud = create_pointcloud_from_depth(
                    intrinsic_matrix=intrinsic_matrices[i],
                    depth=depth_data[i],
                    position=camera_world_pos[i],
                    orientation= camera_world_quat[i],
                    device=self.device,
            )
            # To visualise the point cloud in my Scene
            if i ==0 and visualise_point_cloud and visualise:
                cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
                cfg.markers["hit"].radius = 0.002
                pc_markers = VisualizationMarkers(cfg)
                if pointcloud.size()[0] > 0:
                    pc_markers.visualize(translations=pointcloud)
            if pointcloud.shape[0] > 0:
                # 1. Find the indices of all points that are not part of the floor
                floor_mask = pointcloud[:, 2] > 0.05
                valid_indices = torch.where(floor_mask)[0]

                # 2. Check if we need to downsample these indices
                if valid_indices.shape[0] > 1024:
                    # Randomly shuffle the valid_indices
                    perm = torch.randperm(valid_indices.shape[0], device=self.device)
                    # Select the first 1024 shuffled indices
                    final_indices = valid_indices[perm[:1024]]
                else:
                    # We have fewer than 1024 points, so keep all of them
                    final_indices = valid_indices

                # 3. Use the final list of indices to select points from the original pointcloud
                pointcloud = pointcloud[final_indices]
            else:
                # Ensure pointcloud is an empty tensor if it was empty to begin with
                pointcloud = torch.empty((0, 3), device=self.device)
            point_clouds_list.append(pointcloud.cpu().numpy())
        
        # -- 4. Call the Mapper Update --
        self.occupancy_mapper.update(
            sensor_origins=camera_world_pos.cpu().numpy(),
            point_clouds=point_clouds_list,
        )

        # Update Visibility Map
        cam_data_insp = self._inspection_camera.data
        camera_local_pos_insp = torch.tensor(self.cfg.inspection_camera.offset.pos, device=self.device)
        camera_local_quat_insp = torch.tensor(self.cfg.inspection_camera.offset.rot, device=self.device)
        
        camera_local_quat_insp = camera_local_quat_insp[[3, 0, 1, 2]] # ROS to math convention

        rotated_offsets_insp = quat_apply(robot_quat_w, camera_local_pos_insp.expand_as(robot_pos_w))
        camera_world_pos_insp = robot_pos_w + rotated_offsets_insp
        camera_world_quat_insp = quat_mul(robot_quat_w, camera_local_quat_insp.expand_as(robot_quat_w))
        
        point_clouds_list_insp = []
        depth_data_insp = cam_data_insp.output["distance_to_image_plane"]
        intrinsic_matrices_insp = cam_data_insp.intrinsic_matrices

        for i in range(self.num_envs):

            pointcloud_insp = create_pointcloud_from_depth(
                    intrinsic_matrix=intrinsic_matrices_insp[i],
                    depth=depth_data_insp[i],
                    position=camera_world_pos_insp[i],
                    orientation=camera_world_quat_insp[i],
                    device=self.device,
            )
            if pointcloud_insp.shape[0] > 1024:
                 perm = torch.randperm(pointcloud_insp.shape[0], device=self.device)
                 pointcloud_insp = pointcloud_insp[perm[:1024]]
            point_clouds_list_insp.append(pointcloud_insp.cpu().numpy())
        self.occupancy_mapper.update_visibility(
            sensor_origins=camera_world_pos_insp.cpu().numpy(),
            point_clouds=point_clouds_list_insp,
        )

        # Update Visualizer
        if self.occupancy_mapper.visualizer is not None:
            vis_env_id = self.occupancy_mapper.vis_env_id

            # Get the world pose for that specific robot
            robot_pos_for_vis = self.robot.data.root_pos_w[vis_env_id].cpu().numpy()
            robot_quat_for_vis = self.robot.data.root_quat_w[vis_env_id].cpu().numpy()
            
            # Call the updated visualization method with the pose data
            self.occupancy_mapper.update_visualization(robot_pos_for_vis, robot_quat_for_vis)
  
    def _compute_pose_observation(self) -> torch.Tensor:
        """Compute the robot's pose observation.
        Components:
            - Position
            - Orientation (quaternion)
            - Linear Velocity
            - Angular Velocity
            - Last Action
        """
        pose_world = self.robot.data.root_state_w.clone()

        position = pose_world[...,    :3]
        orientation = pose_world[..., 3:7]
        lin_vel = pose_world[..., 7:10]
        ang_vel = pose_world[..., 10:13]
        action_dim = None
        if isinstance(self.cfg.action_space, gym.spaces.Discrete):
            action_dim = self.cfg.action_space.n
            # One-hot encode the discrete action, ensuring it's long type
            last_action_obs = F.one_hot(self.last_action.squeeze(-1).long(), num_classes=action_dim).float()
        else:
            action_dim = self.last_action.shape[1]
            last_action_obs = self.last_action
            
        obs_buffer = torch.zeros((self.num_envs, 13 + action_dim), device=self.device)

        pos_noise = (torch.rand_like(position) - 0.5) * 0.2
        orientation_noise = (torch.rand_like(orientation) - 0.5) * 0.2
        vel_noise = (torch.randn_like(lin_vel)- 0.5) * 0.2
        ang_vel_noise = (torch.randn_like(ang_vel)-0.5) * 0.2
        action_noise = (torch.randn_like(last_action_obs)-0.5) * 0.2


        obs_buffer[..., :3] = quat_apply_inverse(self.init_quats,  position - self.init_position) + pos_noise
        obs_buffer[..., 3:7] = quat_mul(quat_conjugate(self.init_quats), orientation) + orientation_noise
        obs_buffer[..., 7:10] = quat_apply_inverse(orientation, lin_vel) + vel_noise
        obs_buffer[..., 10:13] = quat_apply_inverse(orientation, ang_vel) + ang_vel_noise

        obs_buffer[..., 13:13 + action_dim] = last_action_obs+ action_noise
        return obs_buffer
    
    def _compute_local_map_observation(self) -> torch.Tensor:
        if not self.cfg.use_occupancy_map:
            raise ValueError("Occupancy map is not enabled in the configuration.")
        robot_pos_w = self.robot.data.root_pos_w
        local_occ_map, local_vis_map = self.occupancy_mapper.get_local_maps(robot_pos_w.cpu().numpy())
        return torch.stack([local_occ_map, local_vis_map], dim=-1).to(self.device)

    def _get_observations(self) -> dict:
        EPSILON = 1e-8
        
        # # Use pure rgb or depth information
        if  "rgb" in self.cfg.observation_camera.data_types:
            front_camera_data = self._obs_camera.data.output[ "rgb"] / 255.0
            if torch.isnan(front_camera_data).any() or torch.isinf(front_camera_data).any():
                print("\n!!! WARNING: Invalid raw data in front camera! Replacing with zeros. !!!\n")
                front_camera_data = torch.zeros_like(front_camera_data)
            
            # normalize the camera data for better training results
            # front_mean = torch.mean(front_camera_data, dim=(1, 2), keepdim=True)
            # front_camera_data -= front_mean

        if   "rgb" in self.cfg.inspection_camera.data_types:
            side_camera_data = self._inspection_camera.data.output["rgb"] / 255.0
            if torch.isnan(side_camera_data).any() or torch.isinf(side_camera_data).any():
                print("\n!!! WARNING: Invalid raw data in side camera! Replacing with zeros. !!!\n")
                side_camera_data = torch.zeros_like(side_camera_data)
            # side_mean = torch.mean(side_camera_data, dim=(1, 2), keepdim=True)
            # side_camera_data -= side_mean
            if torch.isnan(side_camera_data).any():
                raise ValueError("NaN detected in RAW side camera data!")
    
        combined_camera_data = torch.cat([front_camera_data, side_camera_data], dim=-1)
        if torch.isnan(combined_camera_data).any():
            print("\n!!! WARNING: NaN detected in combined camera buffer!!!\n")
            # Uncomment the next line to stop training when a NaN is found
            raise ValueError("NaN detected in combined camera buffer")

        if torch.isinf(combined_camera_data).any():
            print("\n!!! WARNING: Inf detected in combined camera buffer!!!\n")
            # Uncomment the next line to stop training when an Inf is found
            raise ValueError("Inf detected in combined camera buffer")

        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = combined_camera_data.clone()

        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            obs =   {
                'robot-pose': self._compute_pose_observation(),
                'cameras': combined_camera_data.clone(),
                'local-map': self._compute_local_map_observation()
                }
            
            # obs = {
            #     'robot-pose': torch.ones_like(self._compute_pose_observation()) * 0.0,
            #     'cameras': torch.ones_like(combined_camera_data.clone()) * 1.0,
            #     'local-map':  torch.ones_like(self._compute_local_map_observation()) * 2.0
            # }
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            
            obs = (combined_camera_data.clone(), self.robot.data.root_state_w.clone())

        return {"policy": obs}

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        self._update_maps(visualise=debug)

        if self.common_step_counter % self.cfg.logging_interval == 0:
            mean_coverage = np.mean(self.episode_log_buffer["coverage_percent"])
            mean_faces_discovered = np.mean(self.episode_log_buffer["faces_discovered"])
            mean_final_map_entropy = np.mean(self.episode_log_buffer["final_map_entropy"])

            log_data = {
                # Curriculum Status
                "curriculum/inspection_level": self.curriculum.get_inspection_level(),
                "curriculum/spatial_level": self.curriculum.spatial_level,
                "curriculum/success_rate": self.curriculum.success_rate,
                
                # Episode Performance Summary
                "episode_summary/mean_coverage_percent": mean_coverage,
                "episode_summary/mean_faces_discovered": mean_faces_discovered,
                "episode_summary/mean_final_map_entropy": mean_final_map_entropy,

                # Add step-wise rewards (from the cache)
            }
            if self.cached_rewards:
                log_data.update(self.cached_rewards)
            if use_wandb:
                wandb.log(log_data, step=self.common_step_counter)
        return super().step(action)
      
    def _compute_face_discovery_reward_fast(self):
        """
        Compute the reward for discovering new faces.
        """

        segmentation_data = self._inspection_camera.data.output.get("semantic_segmentation")
        face_id_data = self._raycaster_camera.data.output.get("face_ids")

         # Exit if either camera data is missing
        if segmentation_data is None or face_id_data is None:
            return torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        forklift_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            info = self._inspection_camera.data.info[i].get("semantic_segmentation")
            id_to_labels = info.get("idToLabels", {})
            for an_id, label_info in id_to_labels.items():
                if label_info.get("class") == self.cfg.env_parameters["semantics_name"]:
                    forklift_ids[i] = int(an_id)
                    break
        valid_envs_mask = forklift_ids != -1
        forklift_ids_expanded = forklift_ids.view(self.num_envs, 1, 1, 1)

        forklift_mask_all_envs = (segmentation_data == forklift_ids_expanded)
        occlusion_filtered_face_ids = torch.full_like(face_id_data, -1)
        occlusion_filtered_face_ids[forklift_mask_all_envs] = face_id_data[forklift_mask_all_envs]

        face_rewards = torch.zeros(self.num_envs, device=self.device)
        num_faces_inspected = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for env_idx in valid_envs_mask.nonzero(as_tuple=False).squeeze(-1):
            # Get all valid face IDs for this one environment (already computed)
            valid_faces = occlusion_filtered_face_ids[env_idx].flatten()
            valid_faces = valid_faces[valid_faces >= 0]

            if len(valid_faces) > 0:
                existing_faces = self.discovered_faces_buffer[env_idx]
                current_unique_faces = torch.unique(valid_faces)

                combined_faces = torch.cat([existing_faces, current_unique_faces])

                unique_in_combined, counts = torch.unique(combined_faces, return_counts=True)
                newly_discovered_ids = unique_in_combined[counts == 1]
                if newly_discovered_ids.numel() > 0:
                    self.discovered_faces_buffer[env_idx] = torch.cat((self.discovered_faces_buffer[env_idx], newly_discovered_ids))
                    face_rewards[env_idx] = newly_discovered_ids.numel()

            num_faces_inspected[env_idx] = len(self.discovered_faces_buffer[env_idx])
            if debug and env_idx == 0:
                pass
                # self._visualise_faces(face_ids_to_show=occlusion_filtered_face_ids)
        return face_rewards, num_faces_inspected
    
    def _calculate_entropy(self, log_odds):
        """Calculates the Shannon entropy of a map from log-odds."""
        # Clamp to avoid log(0) issues with extreme values
        log_odds = torch.clamp(log_odds, -10.0, 10.0)
        p = torch.sigmoid(log_odds)
        # Use binary_cross_entropy as a numerically stable way to compute entropy
        # H = -(p*log(p) + (1-p)*log(1-p))
        entropy = F.binary_cross_entropy(p, p, reduction='none')
        return entropy
    
    
    def _compute_exploration_rewards(self):
        information_gain_reward = torch.zeros(self.num_envs, device=self.device)
        visibility_increase_reward = torch.zeros(self.num_envs, device=self.device)
        robot_pos_w = self.robot.data.root_pos_w
        local_occ_map, local_vis_map = self.occupancy_mapper.get_local_maps(robot_pos_w.cpu().numpy())
        
        entropy_prev = self._calculate_entropy(self.prev_local_occ_map).sum(dim=(1, 2, 3))
        entropy_new = self._calculate_entropy(local_occ_map).sum(dim=(1, 2, 3))
        information_gain_reward = entropy_prev - entropy_new

        vis_sum_prev = self.prev_local_vis_map.sum(dim=(1, 2, 3))
        vis_sum_new = local_vis_map.sum(dim=(1, 2, 3))
        visibility_increase_reward = torch.relu(vis_sum_new - vis_sum_prev)

        self.prev_local_occ_map = local_occ_map.clone()
        self.prev_local_vis_map = local_vis_map.clone()
        return information_gain_reward, visibility_increase_reward

    def _compute_visitation_reward(self) -> torch.Tensor:
        robot_pos_xy = self.robot.data.root_pos_w[:, :2]  # (num_envs, 2)
        relative_pos = robot_pos_xy - self.visitation_map_origin
        grid_indices = (relative_pos / self.visitation_map_resolution).long()

        grid_x = torch.clamp(grid_indices[:, 0], 0, self.visitation_map_width - 1)
        grid_y = torch.clamp(grid_indices[:, 1], 0, self.visitation_map_height - 1)

        env_indices = torch.arange(self.num_envs, device=self.device)
        visit_counts = self.visitation_maps[env_indices, grid_x, grid_y]

        beta = self.cfg.visitation_beta
        reward =torch.exp(-beta * visit_counts)
        self.visitation_maps[env_indices, grid_x, grid_y] += 1.0
        # Diminishing reward for revisits
        return reward
    def _get_rewards(self) -> torch.Tensor:
        """
            Face Coverage Rewards,
            Exploration Rewards,
            Visibility Rewards
        """
        # return torch.ones(self.num_envs, device=self.device)
        face_discovery_raw, num_faces_inspected = self._compute_face_discovery_reward_fast()
        information_gain_reward, visibility_increase_reward = self._compute_exploration_rewards()

        current_coverage_ratio = num_faces_inspected / self.cfg.max_faces_to_inspect
        coverage_increase_reward = torch.relu(current_coverage_ratio - self.prev_coverage_ratio)

        self.prev_coverage_ratio = current_coverage_ratio.clone()
        visitation_reward = self._compute_visitation_reward()
        success_bonus = torch.where(current_coverage_ratio >= self.curriculum.get_inspection_level(), self.cfg.coverage_reward, 0.0)

        total_reward = (self.cfg.mesh_coverage_reward_scale * face_discovery_raw
                        + self.cfg.information_gain_reward_scale * information_gain_reward
                        + self.cfg.visibility_increase_reward_scale * visibility_increase_reward
                        + visitation_reward
                        + success_bonus
                        + self.cfg.time_penalty
                        )
        # print(f"[DEBUG] Total Reward before scaling: {total_reward}")
        #Make total reward a tensor with float32 dtype
        self.cached_rewards = {
            "reward_components/coverage_increase": coverage_increase_reward.mean().item(),
            "reward_components/info_gain": information_gain_reward.mean().item(),
            "reward_components/visibility_increase": visibility_increase_reward.mean().item(),
            "reward_components/success_bonus": success_bonus.mean().item(),
            "reward_components/total_unscaled": total_reward.mean().item(),
            "reward_components/visitation": visitation_reward.mean().item(),
        }
        total_reward = total_reward.to(torch.float32)
        if torch.isnan(total_reward).any() or torch.isinf(total_reward).any():
            import ipdb; ipdb.set_trace()
            print("NaN or Inf detected in total reward calculation!")

        normalized_reward = self.rewardscaler(total_reward)
        return normalized_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for timeout
        max_length = self.curriculum.get_current_episode_length()
        time_out = self.episode_length_buf >= max_length - 1
        # time_out = self.episode_length_buf >= self.cfg.min_episode_length - 1

        num_faces_inspected = torch.tensor([len(s) for s in self.discovered_faces_buffer], device=self.device)
        coverage_condition = (num_faces_inspected / self.cfg.max_faces_to_inspect) >= self.curriculum.get_inspection_level()
        # if coverage_condition.any() and debug:
        #     print(f"Coverage Condition Met: {coverage_condition.nonzero(as_tuple=False).squeeze(-1)}")

        return coverage_condition, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if len(env_ids)> 0:
            self.prev_coverage_ratio[env_ids] = 0.0 
            self.visitation_maps[env_ids] = 0.0 
            num_faces_inspected = torch.tensor([len(self.discovered_faces_buffer[i]) for i in env_ids], device=self.device)
            coverage_ratio = num_faces_inspected / self.cfg.max_faces_to_inspect
            episode_success = coverage_ratio >= self.curriculum.get_inspection_level()
            self.curriculum.update_inspection_level(episode_success)

            for i, env_id in enumerate(env_ids):
                if debug and env_id == 0:
                    final_face_count = num_faces_inspected[i].item()
                    print(f"--- Episode Summary Env 0 --- Final Faces Discovered: {final_face_count} ---")
                self.episode_log_buffer["coverage_percent"].append(coverage_ratio[i].item() * 100)
                self.episode_log_buffer["faces_discovered"].append(num_faces_inspected[i].item())
                # self.discovered_faces_buffer[env_id].clear()
                self.discovered_faces_buffer[env_id] = torch.tensor([], dtype=torch.float32, device=self.device)

            if self.cfg.use_occupancy_map:
                for env_id in env_ids.cpu().tolist():
                    # Define the slice for the current environment's map
                    start_idx = env_id * self.occupancy_mapper.num_voxels_per_map
                    end_idx = start_idx + self.occupancy_mapper.num_voxels_per_map
                    
                    # Get the log-odds values from the Warp array and convert to a PyTorch tensor
                    env_map_log_odds = wp.to_torch(self.occupancy_mapper.occupancy_map[start_idx:end_idx])
                    
                    # Calculate the total entropy for the map and record it
                    total_entropy = self._calculate_entropy(env_map_log_odds).sum().item()
                    self.episode_log_buffer["final_map_entropy"].append(total_entropy)
        # --- END: New code for entropy logging ---
                self.occupancy_mapper.reset_map(env_ids.cpu().tolist())
                self.prev_local_occ_map[env_ids] = 0.0
                self.prev_local_vis_map[env_ids] = 0.0

        if debug:
            pass
            # for idx in range(len(num_faces_inspected)):
            #    print(f"Number of Faces Discovered: {num_faces_inspected[idx]}")

        super()._reset_idx(env_ids)
            #number of steps taken in the episode
         
        
        # Sample random positions within specified range
        num_resets = len(env_ids)
        new_vel = torch.zeros((num_resets, 3), device=self.device)
        new_pos, new_quat = self.curriculum.get_start_pos(num_resets)
        # new_pos = torch.zeros((num_resets, 3), device=self.device)
        # new_quat = torch.zeros((num_resets, 4), device=self.device)
        # new_quat[:, 0] = 1.0  # No rotation (w, x, y, z)

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