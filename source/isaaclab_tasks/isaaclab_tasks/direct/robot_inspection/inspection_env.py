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
from isaaclab.utils import configclass
from isaacsim.core.utils.semantics import add_labels

from isaaclab.sensors import RayCasterCamera, TiledCamera,  MultiMeshRayCasterCamera
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
from .curriculum_manager import Curriculum
from .utils import  NormalizeReward, visualise_faces
from .occupancy_grid_mapper import OccupancyGridMapper
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import time 
import torch.nn.functional as F
import warp as wp
from collections import defaultdict
# opencv-python-headless-4.11.0.86
from pxr import Usd, UsdGeom, Sdf
from isaaclab.sim.utils import get_current_stage
from .run_config import cfg_mode

# congfig_mode = run_Config
run_cfg = cfg_mode

class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self._ptz_joint_indices, _ = self.robot.find_joints(".*ptz.*")

        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale


        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w


        if self.cfg.mapping_cfg.use_occupancy_map:
            self.occupancy_mapper = OccupancyGridMapper(
                num_envs=self.num_envs,
                map_bounds=self.cfg.mapping_cfg.bounds,
                resolution=self.cfg.mapping_cfg.resolution,
                visibility_surface_hits_only=self.cfg.mapping_cfg.visibility_surface_hits_only,
                visualization_mode = run_cfg.visualisation_mode,
                env_origins= self.scene.env_origins.cpu().numpy(),
                device=self.device,
                # visualize_env_id= None
                visualize_env_id=0 if run_cfg.debug else None
            )
         
        self._setup_tensor_buffers()
        self._setup_camera_zoom()
        self.curriculum = Curriculum(
            num_envs=self.num_envs,
            device=self.device
        )
      
        self.rewardscaler = NormalizeReward(device=self.device)
        self.last_log_step = 0
        self.visualization_timer = 0
        self.visualization_interval = 10

    def close(self):
        """Cleanup for the environment."""
        if self.cfg.mapping_cfg.use_occupancy_map and self.occupancy_mapper.visualizer:
            self.occupancy_mapper.visualizer.close()
        super().close()
        
    def _setup_camera_zoom(self):
        stage = get_current_stage()
        self.camera_prims = []
        self.current_focal_lengths = torch.full(
            (self.num_envs,), 
            self.cfg.robot_phys_cfg.default_focal_length,
            device=self.device,
            dtype=torch.float32
        )

        for i in range(self.num_envs):
            cam_prim_path = self.cfg.sensor_cfg.ptz_camera.prim_path.replace("env_.*", f"env_{i}")
            prim = stage.GetPrimAtPath(cam_prim_path)
            if not prim:
                raise RuntimeError(f"Camera prim not found at path: {cam_prim_path}")
            self.camera_prims.append(UsdGeom.Camera(prim))

    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
        self.init_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_quats = torch.zeros((self.num_envs, 4), device=self.device)

        self.episode_goal_achieved = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if isinstance(self.cfg.action_space, gym.spaces.Discrete):
            action_shape = (self.num_envs, 1)
        else:
            action_shape = (self.num_envs, self.cfg.action_space.shape[0])

        self.last_action = torch.zeros(action_shape, device=self.device)
        self.previous_action_for_rewards = torch.zeros(action_shape, device=self.device)
        self.discovered_faces_buffer = [torch.tensor([], dtype=torch.float32, device=self.device) for _ in range(self.num_envs)]

        self.episode_log_buffer = {
            "coverage_percent": deque(maxlen=self.cfg.scene.num_envs * 4),
            "faces_discovered": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_map_entropy": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_unique_visible_cell_count": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_visited_cells_count":deque(maxlen=self.cfg.scene.num_envs * 4),
            "curriculum/current_threshold": deque(maxlen=self.cfg.scene.num_envs * 4),
        }
        self.reward_logging_buffer = defaultdict(list)
     
        self.success_rate = 0.0
        # Buffers for exploration rewards
        local_map_shape = self.cfg.observation_space["local-map"].shape
        self.prev_local_occ_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_local_vis_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_coverage_ratio = torch.zeros(self.num_envs, device=self.device)
        all_vis_maps = wp.to_torch(self.occupancy_mapper.visibility_map)
        num_cells = all_vis_maps.view(self.num_envs, -1).shape[1]
        initial_score = -float(num_cells)  # Score for a map of all zeros
        self.prev_visibility_score = torch.full((self.num_envs,), initial_score, device=self.device)
        # self.prev_visibility_sum = torch.zeros(self.num_envs, device=self.device)


        if hasattr(self.cfg.mapping_cfg, 'compute_global_map_entropy') and self.cfg.mapping_cfg.compute_global_map_entropy:
            self.prev_global_map_entropy = torch.zeros(self.num_envs, device=self.device)

    

    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self._nav_camera = TiledCamera(self.cfg.sensor_cfg.navigation_camera)
        self.scene.sensors["nav_camera"] = self._nav_camera

        self._ptz_camera = TiledCamera(self.cfg.sensor_cfg.ptz_camera)
        self.scene.sensors["ptz_camera"] = self._ptz_camera

        cfg = sim_utils.UsdFileCfg(
            usd_path="/home/tosin/Documents/GitHub/IsaacLab/assets/ruby.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(density=5.0, mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "inspection_goal")]
            )
        cfg.func(
            "/World/envs/env_.*/rubiks_cube", cfg, 
            translation=(-2, 0.0, 0.02), 
            orientation=(0.70711, 0.0, 0.0, 0.70711)
        )
        
        self.cone = RigidObject(self.cfg.cone_cfg)
        self.scene.rigid_objects["cone"] = self.cone

        self.sphere = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere

        self._raycaster_camera = MultiMeshRayCasterCamera(self.cfg.sensor_cfg.face_raycaster)
        self.scene.sensors["raycaster_camera"] = self._raycaster_camera

        self.scene.clone_environments(copy_from_source=False)
        # clone and replicate
    
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_action_for_rewards.copy_(self.last_action)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.last_action.copy_(actions)
        self.actions = actions.clone()
        
    
    def _apply_action(self) -> None:
        
        if isinstance(self.single_action_space, gym.spaces.Box):
            linear_velocity = self.actions[:, 0] * self.cfg.robot_phys_cfg.max_linear_velocity  # Forward/Backward command
            angular_velocity = self.actions[:, 1] * self.cfg.robot_phys_cfg.max_angular_velocity  # Left/Right turn command


            left_wheel_velocity = (linear_velocity - (angular_velocity * self.cfg.robot_phys_cfg.wheel_separation / 2)) / self.cfg.robot_phys_cfg.wheel_radius
            right_wheel_velocity = (linear_velocity + (angular_velocity * self.cfg.robot_phys_cfg.wheel_separation / 2)) / self.cfg.robot_phys_cfg.wheel_radius


            # Clamp wheel velocities to avoid exceeding max limits
            left_wheel_velocity = torch.clamp(left_wheel_velocity, -self.cfg.robot_phys_cfg.max_wheel_velocity, self.cfg.robot_phys_cfg.max_wheel_velocity)
            right_wheel_velocity = torch.clamp(right_wheel_velocity, -self.cfg.robot_phys_cfg.max_wheel_velocity, self.cfg.robot_phys_cfg.max_wheel_velocity)

            
            self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                        left_wheel_velocity, right_wheel_velocity], dim=1)
            # Position Control for PTZ Camera
            # 90 degrees pi/2 = 1.57, 100 degrees = 1.74533
            #pan_cmd = self.actions[:, 2] * 1.74533
            # 45 degees pi/4 = 0.785
            #tilt_cmd = self.actions[:, 3] * 0.698132
            #ptz_targets = torch.stack([pan_cmd, tilt_cmd], dim=1)

            # Velocity Control for PTZ Camera
            pan_vel_cmd = self.actions[:, 2] * self.cfg.robot_phys_cfg.pan_speed
            tilt_vel_cmd = self.actions[:, 3] * self.cfg.robot_phys_cfg.tilt_speed

            ptz_targets = torch.stack([pan_vel_cmd, tilt_vel_cmd], dim=1)

            # Scale the wheel commands
            wheel_targets = self.wheel_commands * self.cfg.action_scale
            # x = target
            zoom_cmd = self.actions[:, 4]
            self._update_zoom(zoom_cmd)

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
            linear_velocity[actions == 0] = self.cfg.robot_phys_cfg.max_linear_velocity
            angular_velocity[actions == 0] = 0.0

            # Move Forward slow
            linear_velocity[actions == 1] = self.cfg.robot_phys_cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 1] = 0.0

            # Action 2: Turn Left , while moving forward
            linear_velocity[actions == 2] = self.cfg.robot_phys_cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 2] = self.cfg.robot_phys_cfg.max_angular_velocity

            # Action 3: Turn Right
            linear_velocity[actions == 3] = self.cfg.robot_phys_cfg.max_linear_velocity * 0.5
            angular_velocity[actions == 3] = -self.cfg.robot_phys_cfg.max_angular_velocity

            # Action 4: Rotate in Place (Left)
            linear_velocity[actions == 4] = 0.0
            angular_velocity[actions == 4] = self.cfg.robot_phys_cfg.max_angular_velocity

             # Action 4: Rotate in Place (Right)
            linear_velocity[actions == 5] = 0.0
            angular_velocity[actions == 5] = -self.cfg.robot_phys_cfg.max_angular_velocity

            left_wheel_velocity = (linear_velocity - angular_velocity * self.cfg.robot_phys_cfg.wheel_separation / 2.0) / self.cfg.robot_phys_cfg.wheel_radius
            right_wheel_velocity = (linear_velocity + angular_velocity * self.cfg.robot_phys_cfg.wheel_separation / 2.0) / self.cfg.robot_phys_cfg.wheel_radius

            self.wheel_commands = torch.stack([left_wheel_velocity, right_wheel_velocity,
                                       left_wheel_velocity, right_wheel_velocity], dim=1)
            wheel_targets = self.wheel_commands.clone()
            wheel_targets = wheel_targets.view(self.num_envs, -1)

        # print(f"[INFO] Wheel Commands: {self.wheel_commands.clone()}")
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_indices)
        self.robot.set_joint_velocity_target(ptz_targets, joint_ids=self._ptz_joint_indices)

    def _update_zoom(self, zoom_cmd: torch.Tensor):
        delta_zoom = zoom_cmd * self.cfg.robot_phys_cfg.zoom_speed
        self.current_focal_lengths += delta_zoom

        self.current_focal_lengths = torch.clamp(
            self.current_focal_lengths,
            self.cfg.robot_phys_cfg.min_focal_length,
            self.cfg.robot_phys_cfg.max_focal_length
        )
        focal_lengths_cpu = self.current_focal_lengths.cpu().numpy()
        for i, cam_api in enumerate(self.camera_prims):
            # Set the focalLength attribute directly
            # UsdGeom.Camera.GetFocalLengthAttr().Set(value)
            cam_api.GetFocalLengthAttr().Set(float(focal_lengths_cpu[i]))
            
    def _update_maps(self, visualise: bool = False):
        
        if not self.cfg.mapping_cfg.use_occupancy_map:
            return
        
        chassis_min_dist_sq = 0.4 ** 2
        # Update Visitation Map
        self.occupancy_mapper.update_visitation(self.robot.data.root_pos_w.cpu().numpy())

        # Update Occupancy Map
        # ---------------------------------------------------------
        # NAVIGATION CAMERA (Occupancy Map)
        # ---------------------------------------------------------

        nav_cam_pos = self._nav_camera.data.pos_w
        nav_cam_quat = self._nav_camera.data.quat_w_ros

        nav_depth_data = self._nav_camera.data.output["distance_to_image_plane"]
        intrinsic_matrices = self._nav_camera.data.intrinsic_matrices   
        point_clouds_list  = []

        for i in range(self.num_envs):
            pointcloud = create_pointcloud_from_depth(
                    intrinsic_matrix=intrinsic_matrices[i],
                    depth=nav_depth_data[i],
                    position=nav_cam_pos[i],
                    orientation= nav_cam_quat[i],
                    device=self.device,
            )
            
            # To visualise the point cloud in my Scene
            if i ==0 and run_cfg.visualise_point_cloud and visualise:
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
        
        self.occupancy_mapper.update_occupancy(
            sensor_origins= nav_cam_pos.cpu().numpy(),
            point_clouds=point_clouds_list,
        )

        # ---------------------------------------------------------
        # INSPECTION CAMERA (Visibility Map)
        # ---------------------------------------------------------
        insp_cam_pos = self._ptz_camera.data.pos_w
        insp_cam_quat = self._ptz_camera.data.quat_w_ros

        depth_data_insp = self._ptz_camera.data.output["distance_to_image_plane"]
        intrinsic_matrices_insp = self._ptz_camera.data.intrinsic_matrices

        point_clouds_list_insp = []        
        intrinsic_matrices_insp = self._ptz_camera.data.intrinsic_matrices

        for i in range(self.num_envs):
            pointcloud_insp = create_pointcloud_from_depth(
                    intrinsic_matrix=intrinsic_matrices_insp[i],
                    depth=depth_data_insp[i],
                    position=insp_cam_pos[i],
                    orientation=insp_cam_quat[i],
                    device=self.device,
            )
            if pointcloud_insp.shape[0] > 0:
                # Filter out chassis points
                dist_sq_insp = torch.sum((pointcloud_insp - insp_cam_pos[i])**2, dim=1)
                valid_mask_insp = dist_sq_insp > chassis_min_dist_sq
                pointcloud_insp = pointcloud_insp[valid_mask_insp]
            if pointcloud_insp.shape[0] > 1024:
                 perm = torch.randperm(pointcloud_insp.shape[0], device=self.device)
                 pointcloud_insp = pointcloud_insp[perm[:1024]]
            point_clouds_list_insp.append(pointcloud_insp.cpu().numpy())
        self.occupancy_mapper.update_visibility(
            sensor_origins=insp_cam_pos.cpu().numpy(),
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
        position = pose_world[..., :3]
        orientation = pose_world[..., 3:7]
        lin_vel = pose_world[..., 7:10]
        ang_vel = pose_world[..., 10:13]

        if isinstance(self.cfg.action_space, gym.spaces.Discrete):
            action_dim = self.cfg.action_space.n
            # One-hot encode the discrete action, ensuring it's long type
            last_action_obs = F.one_hot(self.last_action.squeeze(-1).long(), num_classes=action_dim).float()
        else:
            action_dim = self.last_action.shape[1]
            last_action_obs = self.last_action
            
        obs_buffer = torch.zeros((self.num_envs, 13 + action_dim+2), device=self.device)

        # pos_noise = (torch.rand_like(position) - 0.5) * 0.2
        # orientation_noise = (torch.rand_like(orientation) - 0.5) * 0.2
        # vel_noise = (torch.randn_like(lin_vel)- 0.5) * 0.2
        # ang_vel_noise = (torch.randn_like(ang_vel)-0.5) * 0.2
        # action_noise = (torch.randn_like(last_action_obs)-0.5) * 0.2

        pos_noise = torch.randn_like(position) * 0.05
        vel_noise = torch.randn_like(lin_vel) * 0.1
        ang_vel_noise = torch.randn_like(ang_vel) * 0.1
        quat_noise = torch.randn_like(orientation) * 0.02

        obs_buffer[..., :3] = quat_apply_inverse(self.init_quats,  position - self.init_position) + pos_noise
        rel_quat = quat_mul(quat_conjugate(self.init_quats), orientation) + quat_noise
        obs_buffer[..., 3:7] = F.normalize(rel_quat, p=2, dim=-1)

        # obs_buffer[..., 3:7] = quat_mul(quat_conjugate(self.init_quats), orientation) + orientation_noise
        obs_buffer[..., 7:10] = quat_apply_inverse(orientation, lin_vel) + vel_noise
        obs_buffer[..., 10:13] = quat_apply_inverse(orientation, ang_vel) + ang_vel_noise

        obs_buffer[..., 13:13 + action_dim] = last_action_obs

        ptz_joint_pos = self.robot.data.joint_pos[:, self._ptz_joint_indices]
        # print(f"PTZ Joint Positions: {ptz_joint_pos}")
        obs_buffer[..., 13 + action_dim : 13 + action_dim + 2] = ptz_joint_pos

        if torch.isnan(obs_buffer).any():
            print("\n[ENV DEBUG] NaN detected in _compute_pose_observation!")
            print(f"  Position has NaNs: {torch.isnan(position).any().item()}")
            print(f"  Orientation has NaNs: {torch.isnan(orientation).any().item()}")
            print(f"  Lin Vel has NaNs: {torch.isnan(lin_vel).any().item()}")
            print(f"  Ang Vel has NaNs: {torch.isnan(ang_vel).any().item()}")
            print(f"  Last Action has NaNs: {torch.isnan(last_action_obs).any().item()}")
        return obs_buffer
    
    def _compute_local_map_observation(self) -> torch.Tensor:
        if not self.cfg.mapping_cfg.use_occupancy_map:
            raise ValueError("Occupancy map is not enabled in the configuration.")
        robot_pos_w = self.robot.data.root_pos_w
        local_occ_map, local_vis_map, local_visit = self.occupancy_mapper.get_local_maps(robot_pos_w.cpu().numpy())
        for name, tensor in {"occ": local_occ_map, "vis": local_vis_map, "visit": local_visit}.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                msg = f"Invalid value (NaN or Inf) in Local Map: {name}"
                print(msg)
                raise ValueError(msg)
        return torch.stack([local_occ_map, local_vis_map, local_visit], dim=-1).to(self.device)

    def _get_observations(self) -> dict:
        # return  {"policy": None}
        front_camera_data = torch.empty(0, device=self.device)
        ptz_camera_data = torch.empty(0, device=self.device)

        if  "rgb" in self.cfg.sensor_cfg.navigation_camera.data_types:
            front_camera_data = self._nav_camera.data.output[ "rgb"] / 255.0
            if torch.isnan(front_camera_data).any():
                print("[ENV DEBUG] NaN detected in Front Camera Data!")

        if  "rgb" in self.cfg.sensor_cfg.ptz_camera.data_types:
            ptz_camera_data = self._ptz_camera.data.output["rgb"] / 255.0
            if torch.isnan(ptz_camera_data).any():
                print("[ENV DEBUG] NaN detected in PTZ Camera Data!")
        semantic_channel = torch.zeros(
            (self.num_envs, self.cfg.sensor_cfg.ptz_camera.height, self.cfg.sensor_cfg.ptz_camera.width, 1),
            device=self.device
        )
        if  "semantic_segmentation" in self.cfg.sensor_cfg.ptz_camera.data_types:
            target_mask = self._get_semantic_mask(self._ptz_camera)
            
            if target_mask is not None:
                semantic_channel = target_mask.float()

                if torch.isnan(semantic_channel).any():
                    print("[ENV DEBUG] NaN detected in Semantic Channel (Target Mask)!")
            
        combined_camera_data = torch.cat([
            front_camera_data, ptz_camera_data, semantic_channel], dim=-1)
        pose_obs = self._compute_pose_observation()
        map_obs = self._compute_local_map_observation()
        if torch.isinf(pose_obs).any() or torch.isnan(pose_obs).any():
            print("\n[CRITICAL FAILURE] Infinite or NaN detected in ROBOT POSE")
            print(f"  > Min Value: {pose_obs.min().item()}")
            print(f"  > Max Value: {pose_obs.max().item()}")
            print(f"  > Robot Velocity (First Env): {self.robot.data.root_lin_vel_w[0]}")
            raise ValueError("Training stopped: Robot Physics Exploded (Pose contains Inf/NaN)")
        if torch.isinf(map_obs).any() or torch.isnan(map_obs).any():
            print("\n[CRITICAL FAILURE] Infinite or NaN detected in LOCAL MAP")
            print(f"  > Min Value: {map_obs.min().item()}")
            print(f"  > Max Value: {map_obs.max().item()}")
            raise ValueError("Training stopped: Local Map contains Inf/NaN (likely divide by zero in mapper)")

        # Check Cameras (Rendering Issues)
        if torch.isinf(combined_camera_data).any() or torch.isnan(combined_camera_data).any():
            print("\n[CRITICAL FAILURE] Infinite or NaN detected in CAMERAS")
            raise ValueError("Training stopped: Camera tensor contains Inf/NaN")
        
        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = combined_camera_data.clone()

        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            obs =   {
                'robot-pose': pose_obs,
                'cameras': combined_camera_data.clone(),
                'local-map': map_obs
                }
            
            # obs = {
            #     'robot-pose': torch.ones_like(self._compute_pose_observation()) * 0.0,
            #     'cameras': torch.ones_like(combined_camera_data.clone()) * 1.0,
            #     'local-map':  torch.ones_like(self._compute_local_map_observation()) * 2.0
            # }
        return {"policy": obs}

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:

        if self.common_step_counter % self.cfg.mapping_cfg.map_update_interval == 0:
            self._update_maps(visualise=run_cfg.debug)

        if not run_cfg.debug and self.common_step_counter % self.cfg.logging_interval == 0:
            mean_coverage = np.mean(self.episode_log_buffer["coverage_percent"])
            mean_faces_discovered = np.mean(self.episode_log_buffer["faces_discovered"])
            mean_final_map_entropy = np.mean(self.episode_log_buffer["final_map_entropy"])
            mean_final_visited_cells_count = np.mean(self.episode_log_buffer["final_visited_cells_count"])
            mean_final_unique_visible_cell_count = np.mean(self.episode_log_buffer["final_unique_visible_cell_count"])
            mean_current_threshold = np.mean(self.episode_log_buffer["curriculum/current_threshold"])

            # mean_current_episode_length = np.mean(self.episode_log_buffer["curriculum/current_episode_length"])
            log_data = {
                # Curriculum Status
                "curriculum/exploration_threshold_goal": mean_current_threshold,
                # "curriculum/episode_length_limit": mean_current_episode_length,
                "curriculum/success_rate": self.curriculum.success_rate,
                
                # Episode Performance Summary
                "episode_summary/mean_coverage_percent": mean_coverage,
                "episode_summary/mean_faces_discovered": mean_faces_discovered,
                "episode_summary/mean_final_map_entropy": mean_final_map_entropy,
                "episode_summary/mean_final_unique_visible_cell_count": mean_final_unique_visible_cell_count,
                "episode_summary/mean_final_visited_cells_count": mean_final_visited_cells_count,

            }
            for reward_name, reward_values in self.reward_logging_buffer.items():
                if len(reward_values) > 0:
                    log_data[reward_name] = np.mean(reward_values)
            # Clear the reward logging buffer after logging
            self.reward_logging_buffer.clear()
            if run_cfg.use_wandb:
                # wandb.log(log_data, step=self.common_step_counter)
                wandb.log(log_data)

        return super().step(action)
      
    def _get_semantic_mask(self, camera) -> torch.Tensor | None:
        """
        Utility to extract the binary mask for the target object from a given camera.
        Returns a boolean tensor (N, H, W, 1) or None if target not found.
        """
        # Ensure data exists
        seg_data = camera.data.output["semantic_segmentation"]
        
        # Retrieve label mapping
        info = camera.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        target_class_name = self.cfg.env_parameters["semantics_name"]

        # Find the ID associated with the class name
        target_id = None
        for k, v in id_to_labels.items():
            if v.get("class") == target_class_name:
                target_id = int(k)
                break
        
        if target_id is not None:
            # Return boolean mask (N, H, W, 1)
            return seg_data == target_id
        
        return None
    
    def _compute_face_discovery_reward_fast(self):
        """
        Compute the reward for discovering new faces.
        """


        face_ids = self._raycaster_camera.data.output.get("face_ids")
        target_mask = self._get_semantic_mask(self._ptz_camera)

         # Exit if either camera data is missing
        if target_mask is None or face_ids is None:
            return (torch.zeros(self.num_envs, device=self.device), 
                    torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
    
    
        occlusion_filtered_face_ids = torch.full_like(face_ids, -1)
        occlusion_filtered_face_ids[target_mask] = face_ids[target_mask]

        # per environments operation
        face_rewards = torch.zeros(self.num_envs, device=self.device)
        num_faces_inspected = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for env_idx in range(self.num_envs):
            # Get all valid face IDs for this one environment (already computed)
            valid_faces = occlusion_filtered_face_ids[env_idx].flatten()
            valid_faces = valid_faces[valid_faces >= 0]

            if valid_faces.numel() > 0:
                existing_faces = self.discovered_faces_buffer[env_idx]
                current_unique_faces = torch.unique(valid_faces)

                is_new_mask = ~torch.isin(current_unique_faces, self.discovered_faces_buffer[env_idx])
                newly_discovered_ids = current_unique_faces[is_new_mask]

                # combined_faces = torch.cat([existing_faces, current_unique_faces])
                # unique_in_combined, counts = torch.unique(combined_faces, return_counts=True)
                # newly_discovered_ids = unique_in_combined[counts == 1]

                if newly_discovered_ids.numel() > 0:
                    self.discovered_faces_buffer[env_idx] = torch.cat(
                        (self.discovered_faces_buffer[env_idx], newly_discovered_ids)
                    )
                    face_rewards[env_idx] = newly_discovered_ids.numel()

            num_faces_inspected[env_idx] = self.discovered_faces_buffer[env_idx].numel()
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
        # --- Entropy / Information Gain ---
        all_maps_log_odds = wp.to_torch(self.occupancy_mapper.occupancy_map)
        all_maps_log_odds = all_maps_log_odds.view(self.num_envs, -1)
        current_entropy = self._calculate_entropy(all_maps_log_odds).sum(dim=1)
        information_gain = torch.relu(self.prev_global_map_entropy - current_entropy)
        self.prev_global_map_entropy = current_entropy.clone()

        # ---Surface Visibility Increase ---
        k = self.cfg.reward_cfg.visibility_decay_factor
        all_vis_maps = wp.to_torch(self.occupancy_mapper.visibility_map)
        all_vis_maps = all_vis_maps.view(self.num_envs, -1)
        current_visibility_score = -torch.exp(-k * all_vis_maps).sum(dim=1)
        visibility_increase_reward = torch.relu(current_visibility_score - self.prev_visibility_score)
        self.prev_visibility_score = current_visibility_score.clone()

        return information_gain, visibility_increase_reward

    def _compute_visitation_reward(self) -> torch.Tensor:
        """
        Computes a reward for visiting new voxels, based on the 3D visitation map.
        The reward diminishes exponentially as a voxel is revisited.
        """
        robot_pos_w = self.robot.data.root_pos_w    # (num_envs, 2)
        map_origins = torch.from_numpy(self.occupancy_mapper.world_map_origins).to(self.device)
        voxel_size = self.occupancy_mapper.resolution
        map_dims = self.occupancy_mapper.map_dims

        relative_pos = robot_pos_w - map_origins
        

        grid_indices =(relative_pos / voxel_size).long()
        ix, iy, iz = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]
        valid_mask = (ix >= 0) & (ix < map_dims[0]) & \
                     (iy >= 0) & (iy < map_dims[1]) & \
                     (iz >= 0) & (iz < map_dims[2])

        # Initialize reward tensor
        reward = torch.zeros(self.num_envs, device=self.device)
        if not valid_mask.any():
            return reward

        # Calculate linear indices for valid positions
        linear_indices = ix[valid_mask] * map_dims[1] * map_dims[2] + \
                         iy[valid_mask] * map_dims[2] + \
                         iz[valid_mask]
        
        # Add environment-specific offset to get the global index
        env_ids = torch.arange(self.num_envs, device=self.device)[valid_mask]
        map_offset = env_ids * self.occupancy_mapper.num_voxels_per_map
        global_indices = map_offset + linear_indices

        # Get visitation counts from the GPU map
        all_visit_counts_torch = wp.to_torch(self.occupancy_mapper.visitation_map)
        current_counts = all_visit_counts_torch[global_indices]

        # Calculate reward: R = exp(-beta * N), where N is the visit count
        # Note: We subtract 1.0 because the map was already updated this step.
        # We want to reward based on the state *before* the current visit.
        reward[valid_mask] = torch.exp(-self.cfg.reward_cfg.visitation_decay_factor * (current_counts - 1.0))
        
        return reward
    
    def _get_rewards(self) -> torch.Tensor:
        """
            Face Coverage Rewards,
            Exploration Rewards,
            Visibility Rewards
        """
        # return torch.zeros(self.num_envs, device=self.device)
        face_discovery_raw, total_num_faces_inspected = self._compute_face_discovery_reward_fast()
        information_gain_reward, visibility_increase_reward = self._compute_exploration_rewards()
        visitation_reward = self._compute_visitation_reward()
        action_delta = torch.sum(torch.square(self.actions - self.previous_action_for_rewards), dim=1)
        # Camera Penalty pan and tilt
        camera_delta = torch.sum(
            torch.square(self.actions[:, 2:4] - self.previous_action_for_rewards[:, 2:4]
        ), dim=1)

        # Inpsection Coverage Ratio and Success Bonus
        current_coverage_ratio = total_num_faces_inspected / self.cfg.max_faces_to_inspect
        self.prev_coverage_ratio = current_coverage_ratio.clone()
       
        success_bonus = torch.where(
            current_coverage_ratio >= self.curriculum.get_current_coverage_goal(), 
            self.cfg.reward_cfg.coverage_reward,
            0.0
        )

        total_reward = (self.cfg.reward_cfg.mesh_coverage_reward_scale * face_discovery_raw
                        + self.cfg.reward_cfg.information_gain_reward_scale * information_gain_reward
                        + self.cfg.reward_cfg.visibility_increase_reward_scale * visibility_increase_reward
                        #+ self.cfg.reward_cfg.visitation_reward_scale * visitation_reward # Added visitation reward
                        # + self.cfg.reward_cfg.action_penalty_scale * action_delta
                        # + self.cfg.reward_cfg.ptz_penalty_scale * camera_delta
                        + success_bonus
                        + self.cfg.reward_cfg.time_penalty
                        )
        self._cache_rewards(
            face_discovery_raw,
            information_gain_reward, 
            visibility_increase_reward, 
            action_delta,
            camera_delta,
            total_reward
            )
        # print(f"[DEBUG] Total Reward before scaling: {total_reward}")
        # Logging
      
        total_reward = total_reward.to(torch.float32)
        if torch.isnan(total_reward).any() or torch.isinf(total_reward).any():
            import ipdb; ipdb.set_trace()
            print("NaN or Inf detected in total reward calculation!")
            raise ValueError("Training stopped: NaN or Inf detected in total reward calculation!")
        normalized_reward = self.rewardscaler(total_reward)
        return normalized_reward
    
    def _cache_rewards(self, face_discovery, info_gain, visibility_increase, action_delta, camera_delta, total_unscaled):
        step_face_discovery = face_discovery.mean().item()
        step_info_gain = info_gain.mean().item()
        step_vis_increase = visibility_increase.mean().item()
        step_action_delta = action_delta.mean().item()
        step_camera_delta = camera_delta.mean().item()
        step_total_raw = total_unscaled.mean().item()

        self.reward_logging_buffer["reward_components/face_discovery"].append(step_face_discovery)
        self.reward_logging_buffer["reward_components/info_gain"].append(step_info_gain)
        self.reward_logging_buffer["reward_components/visibility_increase"].append(step_vis_increase)
        self.reward_logging_buffer["reward_components/total_unscaled"].append(step_total_raw)
        self.reward_logging_buffer["reward_components/action_penalty"].append(step_action_delta)
        self.reward_logging_buffer["reward_components/camera_penalty"].append(step_camera_delta)
        
        # Scaled versioons
        self.reward_logging_buffer["reward_components/face_discovery_scaled"].append(   
            self.cfg.reward_cfg.mesh_coverage_reward_scale * step_face_discovery
        )
        self.reward_logging_buffer["reward_components/info_gain_scaled"].append(
            self.cfg.reward_cfg.information_gain_reward_scale * step_info_gain
        )
        self.reward_logging_buffer["reward_components/visibility_increase_scaled"].append(
            self.cfg.reward_cfg.visibility_increase_reward_scale * step_vis_increase
        )
        self.reward_logging_buffer["reward_components/action_penalty_scaled"].append(
            self.cfg.reward_cfg.action_penalty_scale * step_action_delta
        )
        self.reward_logging_buffer["reward_components/camera_penalty_scaled"].append(
            self.cfg.reward_cfg.ptz_penalty_scale * step_camera_delta
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for timeout
        time_out = self.episode_length_buf >= self.cfg.min_episode_length - 1
        num_faces_inspected = torch.tensor([len(s) for s in self.discovered_faces_buffer], device=self.device)
        coverage_condition = (num_faces_inspected / self.cfg.max_faces_to_inspect) >= self.curriculum.get_current_coverage_goal()
        return coverage_condition, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if len(env_ids)> 0:
            self.last_action[env_ids] = 0.0
            self.previous_action_for_rewards[env_ids] = 0.0
            num_faces_inspected = torch.tensor([len(self.discovered_faces_buffer[i]) for i in env_ids], device=self.device)
            achieved_coverage_ratios = num_faces_inspected / float(self.cfg.max_faces_to_inspect)
            current_goal = self.curriculum.get_current_coverage_goal()
            episode_successes = achieved_coverage_ratios >= current_goal
            self.curriculum.update_curriculum(episode_successes)

            # Logging
            for i, env_id in enumerate(env_ids):
                if run_cfg.debug and env_id == 0:
                    final_face_count = num_faces_inspected[i].item()
                    print(f"--- Episode Summary Env 0 --- Final Faces Discovered: {final_face_count} ---")

                self.episode_log_buffer["coverage_percent"].append(achieved_coverage_ratios[i].item() * 100)
                self.episode_log_buffer["faces_discovered"].append(num_faces_inspected[i].item())
                self.episode_log_buffer["curriculum/current_threshold"].append(current_goal)

                # Clear Buffers
                self.discovered_faces_buffer[env_id] = torch.tensor([], dtype=torch.float32, device=self.device)
                self.prev_coverage_ratio[env_id] = 0.0

            # Map Logging and Reset
            if self.cfg.mapping_cfg.use_occupancy_map:

                all_vis_maps_torch = wp.to_torch(self.occupancy_mapper.visibility_map).view(self.num_envs, -1)
                all_occ_maps_torch = wp.to_torch(self.occupancy_mapper.occupancy_map).view(self.num_envs, -1)
                all_visit_maps_torch = wp.to_torch(self.occupancy_mapper.visitation_map).view(self.num_envs, -1) # NEW

                final_vis_sums = all_vis_maps_torch.sum(dim=1)
                final_entropies = self._calculate_entropy(all_occ_maps_torch).sum(dim=1)
                final_robot_path_cells = (all_visit_maps_torch > 0).sum(dim=1)
                final_unique_visible_cells = (all_vis_maps_torch > 0).sum(dim=1)

                current_scores = final_unique_visible_cells[env_ids]



                for env_id in env_ids.cpu().tolist():
                    # Covergae for Trajectory
                    self.episode_log_buffer["final_visited_cells_count"].append(final_robot_path_cells[env_id].item())
                    #Surface Coverage Metric
                    self.episode_log_buffer["final_unique_visible_cell_count"].append(final_unique_visible_cells[env_id].item())
                    # Map Entropy
                    self.episode_log_buffer["final_map_entropy"].append(final_entropies[env_id].item())
                    if run_cfg.debug and env_id == 0:
                        print(f"--- Episode Summary Env 0 --- Final Unique Visible Cells: {final_unique_visible_cells[env_id].item()} ---")
                        print(f"--- Episode Summary Env 0 --- Final Map Entropy: {final_entropies[env_id].item()} ---")


        # --- END: New code for entropy logging ---
                self.occupancy_mapper.reset_map(env_ids.cpu().tolist())
                all_vis_maps = wp.to_torch(self.occupancy_mapper.visibility_map)
                num_cells = all_vis_maps.view(self.num_envs, -1).shape[1]
                initial_score = -float(num_cells)
                self.prev_visibility_score[env_ids] = initial_score 
                all_maps_log_odds = wp.to_torch(self.occupancy_mapper.occupancy_map)
                all_maps_log_odds = all_maps_log_odds.view(self.num_envs, -1)
                initial_entropy = self._calculate_entropy(all_maps_log_odds).sum(dim=1)
                self.prev_global_map_entropy[env_ids] = initial_entropy[env_ids]
                
                self.prev_local_occ_map[env_ids] = 0.0
                self.prev_local_vis_map[env_ids] = 0.0


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