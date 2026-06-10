# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
from __future__ import annotations

from collections import deque
import os
import gymnasium as gym
import torch
from collections.abc import Sequence
import numpy as np

from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaacsim.core.utils.semantics import add_labels

from isaaclab.sensors import RayCasterCamera, TiledCamera,  MultiMeshRayCasterCamera
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
import isaacsim.core.utils.stage as stage_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG, CUBOID_MARKER_CFG
# from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul, quat_apply, quat_conjugate, quat_apply_inverse, yaw_quat
# from semanc_manager import SemanticManager, add_semantic_tags_from_config
from .inspection_cfg import Isaac3dinspectionEnvCfg
from .spatial_state_manager import SpatialStateManager
# import wandb
from .curriculum_manager import Curriculum
from .utils import  NormalizeReward, visualise_faces, _show_face_ids_
from .inspection_logger import InspectionLogger
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import time 
import torch.nn.functional as F
import warp as wp
from collections import defaultdict
# opencv-python-headless-4.11.0.86
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, PhysxSchema, Gf
from isaaclab.sim.utils import get_current_stage
from .run_config import cfg_mode
from .utils.data_collector import DataCollector
from .utils.reconstruction_data_collector import ReconstructionDataCollector

# congfig_mode = run_Config
run_cfg = cfg_mode

class Isaac3dinspectionEnv(DirectRLEnv):
    cfg: Isaac3dinspectionEnvCfg

    def __init__(self, cfg: Isaac3dinspectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
        self._wheel_joint_indices, self._wheel_joint_names = self.robot.find_joints(".*wheel.*")
        self._ptz_joint_indices, _ = self.robot.find_joints(".*ptz.*")

        self.wheel_velocity_scale = self.cfg.wheel_velocity_scale

        # Assign targets
        target_keys = list(self.cfg.inspection_goal_cfg.inspection_targets.keys())
        self.env_target_names = np.random.choice(target_keys, self.num_envs)
        
        # Setup specific faces for environments
        self.total_mesh_faces = torch.zeros(self.num_envs, device=self.device)
        for i, target_name in enumerate(self.env_target_names):
            faces = self.cfg.inspection_goal_cfg.inspection_targets[target_name].num_faces
            self.total_mesh_faces[i] = faces

        self.robot_pos = self.robot.data.root_pos_w
        self.robot_vel = self.robot.data.root_lin_vel_w


        if self.cfg.mapping_cfg.use_occupancy_map:
            self.map_manager = SpatialStateManager(
                num_envs=self.num_envs,
                map_bounds=self.cfg.mapping_cfg.bounds,
                resolution=self.cfg.mapping_cfg.resolution,
                visibility_surface_hits_only=self.cfg.mapping_cfg.visibility_surface_hits_only,
                local_map_dims=self.cfg.mapping_cfg.local_map_dims,
                egocentric_map=getattr(self.cfg.mapping_cfg, "egocentric_map", True),
                log_odds_free=getattr(self.cfg.mapping_cfg, "log_odds_free", -0.4),
                log_odds_occupied=getattr(self.cfg.mapping_cfg, "log_odds_occupied", 0.8),
                clamp_min=getattr(self.cfg.mapping_cfg, "clamp_min", -5.0),
                clamp_max=getattr(self.cfg.mapping_cfg, "clamp_max", 5.0),
                visualization_mode = run_cfg.visualisation_mode,
                env_origins= self.scene.env_origins.cpu().numpy(),
                device=self.device,
                # visualize_env_id= None
                visualize_env_id=0 if (run_cfg.debug and getattr(run_cfg, 'enable_voxel_visualization', False)) else None
            )
        self.curriculum = Curriculum(
            num_envs=self.num_envs,
            device=self.device,
    
        )

        # Scale spatial rewards dynamically based on map resolution (computed once at init)
        res_ratio = self.cfg.mapping_cfg.resolution / 0.25
        self.cfg.reward_cfg.information_gain_reward_scale *= (res_ratio ** 3)
        self.cfg.reward_cfg.visibility_increase_reward_scale *= (res_ratio ** 2)
        self.cfg.reward_cfg.visitation_reward_scale *= res_ratio

        self._setup_tensor_buffers()
        self._setup_camera_zoom()
        
        # Initialize point cloud markers if needed
        self.pc_markers = None
        if run_cfg.visualise_point_cloud:
            cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            cfg.markers["hit"].radius = 0.002
            self.pc_markers = VisualizationMarkers(cfg)
      
        self.rewardscaler = NormalizeReward(device=self.device)
        self.last_log_step = 0
        self.visualization_timer = 0
        self.visualization_interval = 10
        self.visualization_interval = 10
        self.face_max_counts = torch.zeros(self.q_capacity, dtype=torch.long, device=self.device)
        self.global_max_ray_count = torch.zeros(1, dtype=torch.long, device=self.device)
        
        self.spawn_markers = None



        # Initialize Data Collector if needed (saves RGB/images)
        self.data_collector = None
        if hasattr(run_cfg, "data_recording_path") and getattr(run_cfg, "save_images", False):
            self.data_collector = DataCollector(run_cfg, device=self.device)

        # Initialize Point Cloud Collector if needed (saves Depth/masks)
        self.pc_collector = None
        
        if hasattr(run_cfg, "data_recording_path") and getattr(run_cfg, "save_depth", False):
             self.pc_collector = ReconstructionDataCollector(run_cfg, device=self.device)


    def close(self):
        """Cleanup for the environment."""
        if self.cfg.mapping_cfg.use_occupancy_map and self.map_manager.visualizer:
            self.map_manager.visualizer.close()
            
        if hasattr(self, 'pc_collector') and self.pc_collector:
            try:
                # Try to safely get the latest max distance / faces for Env 0
                max_dist = self.max_distance_reached[0].item() if hasattr(self, 'max_distance_reached') else 0.0
                
                # Best attempt at getting the max faces discovered so far in this current episode
                faces = 0
                if hasattr(self, 'best_q_per_face'):
                    faces = (self.best_q_per_face[0] > 0.0).sum().item()
                    
                self.pc_collector.flush_if_best(faces, max_dist)
            except Exception as e:
                print(f"[ERROR] Could not save summary during close: {e}")

        if hasattr(self, 'data_collector') and self.data_collector:
            self.data_collector.save() # Just prints final stats now
            
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
        self.committed_focal_lengths = self.current_focal_lengths.clone()

        self.high_res_camera_prims = []
        for i in range(self.num_envs):
            cam_prim_path = self.cfg.sensor_cfg.ptz_camera.prim_path.replace("env_.*", f"env_{i}")
            prim = stage.GetPrimAtPath(cam_prim_path)
            if not prim:
                raise RuntimeError(f"Camera prim not found at path: {cam_prim_path}")
            self.camera_prims.append(UsdGeom.Camera(prim))
            
            if getattr(run_cfg, "add_high_res_inspection_camera", False) and hasattr(self.cfg.sensor_cfg, "high_res_ptz_camera"):
                hr_cam_prim_path = self.cfg.sensor_cfg.high_res_ptz_camera.prim_path.replace("env_.*", f"env_{i}")
                hr_prim = stage.GetPrimAtPath(hr_cam_prim_path)
                if hr_prim:
                    self.high_res_camera_prims.append(UsdGeom.Camera(hr_prim))

    def _setup_tensor_buffers(self):
        """Pre-allocate all tensors to avoid memory allocation during runtime."""
        self.init_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_quats = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Allocate enough capacity to avoid re-allocating dynamically on the CPU
        # USD mesh Face IDs can be sparse/non-contiguous, requiring large array bounds
        max_faces = int(self.total_mesh_faces.max().item())
        self.q_capacity = max(1_700_000, max_faces + 1000)
        self.best_q_per_face = torch.zeros((self.num_envs, self.q_capacity), device=self.device, dtype=torch.float32)

        self.episode_goal_achieved = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if isinstance(self.cfg.action_space, gym.spaces.Discrete):
            action_shape = (self.num_envs, 1)
        else:
            action_shape = (self.num_envs, self.cfg.action_space.shape[0])

        self.last_action = torch.zeros(action_shape, device=self.device)
        self.previous_action_for_rewards = torch.zeros(action_shape, device=self.device)
        
        # Buffers are managed by the logger
        self.logger = InspectionLogger(self.cfg, use_wandb=run_cfg.use_wandb, debug=run_cfg.debug, window_size=self.curriculum.success_buffer.maxlen)
        self.episode_log_buffer = self.logger.episode_log_buffer
        self.reward_logging_buffer = self.logger.reward_logging_buffer


        self.success_rate = 0.0
        # Buffers for exploration rewards
        local_map_shape = self.cfg.observation_space["local-map"].shape
        self.prev_local_occ_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_local_vis_map = torch.zeros((self.num_envs, *local_map_shape[:-1]), device=self.device)
        self.prev_coverage_ratio = torch.zeros(self.num_envs, device=self.device)
        all_vis_maps = wp.to_torch(self.map_manager.visibility_map)
        num_cells = all_vis_maps.view(self.num_envs, -1).shape[1]
        initial_score = -float(num_cells)  # Score for a map of all zeros
        self.prev_visibility_score = torch.full((self.num_envs,), initial_score, device=self.device)
        # self.prev_visibility_sum = torch.zeros(self.num_envs, device=self.device)


        if hasattr(self.cfg.mapping_cfg, 'compute_global_map_entropy') and self.cfg.mapping_cfg.compute_global_map_entropy:
            self.prev_global_map_entropy = torch.zeros(self.num_envs, device=self.device)
        if hasattr(run_cfg, 'data_recording_path') and run_cfg.data_recording_path:
            self.max_distance_reached = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        #Add robot, camera and terain to the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self._nav_camera = TiledCamera(self.cfg.sensor_cfg.navigation_camera)
        self.scene.sensors["nav_camera"] = self._nav_camera

        self._ptz_camera = TiledCamera(self.cfg.sensor_cfg.ptz_camera)
        self.scene.sensors["ptz_camera"] = self._ptz_camera

        if getattr(run_cfg, "add_high_res_inspection_camera", False) and hasattr(self.cfg.sensor_cfg, "high_res_ptz_camera"):
            self._high_res_ptz_camera = TiledCamera(self.cfg.sensor_cfg.high_res_ptz_camera)
            self.scene.sensors["high_res_ptz_camera"] = self._high_res_ptz_camera

        # --- RESTORED MANUAL SPAWN LOGIC ---
        stage = get_current_stage()
        import re
        
        # Clone heavily populated structures immediately to build parallel env branches
        
        self.inspection_goals = {}
        self.goal_prims_dict = {}

        for target_name, target_cfg in self.cfg.inspection_goal_cfg.inspection_targets.items():
            usd_path = target_cfg.usd_path
            prim_path_template = target_cfg.prim_path
            scale = target_cfg.scale
            orientation = target_cfg.orientation

            obj_cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=scale,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(density=5.0, mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                semantic_tags=[(self.cfg.inspection_goal_cfg.semantics_type, target_name)]
            )
            obj_cfg.func(
                prim_path_template,
                obj_cfg,
                translation=(2.0, 0.0, 0.4),
                orientation=orientation,
            )

            # Force collision
            self.goal_prims_dict[target_name] = []
            for i in range(self.num_envs):
                if "env_.*" in prim_path_template:
                    obj_prim_path = prim_path_template.replace("env_.*", f"env_{i}")
                else:
                    obj_prim_path = f"{prim_path_template}_{i}" # fallback
                
                obj_prim = stage.GetPrimAtPath(obj_prim_path)
                if obj_prim.IsValid():
                    if not obj_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI.Apply(obj_prim)

                    for child in Usd.PrimRange(obj_prim):
                        if child.IsA(UsdGeom.Mesh):
                            if not child.HasAPI(UsdPhysics.CollisionAPI):
                                UsdPhysics.CollisionAPI.Apply(child)
                            if not child.HasAPI(UsdPhysics.MeshCollisionAPI):
                                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(child)
                                mesh_collision.GetApproximationAttr().Set("convexHull")
                            if not child.HasAPI(PhysxSchema.PhysxCollisionAPI):
                                PhysxSchema.PhysxCollisionAPI.Apply(child)
                                
                    self.goal_prims_dict[target_name].append(UsdGeom.Xformable(obj_prim))
                else:
                    print(f"[WARNING] Goal prim not found at {obj_prim_path}")
                    self.goal_prims_dict[target_name].append(None)
                    
            # Create RigidObject wrapper
            self.inspection_goals[target_name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path_template,
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.4))
                )
            )
            self.scene.rigid_objects[f"inspection_goal_{target_name}"] = self.inspection_goals[target_name]
        # Spawn obstacles dynamically
        from .utils.dataset_handler import ObstacleDatasetHandler
        max_obstacles = self.cfg.max_obstacles
        if getattr(run_cfg, "is_simplified", False):
            max_obstacles = 0
            
        handler = ObstacleDatasetHandler(max_obstacles=max_obstacles)
        obstacle_cfgs = handler.get_obstacle_configs()
        
        self.obstacles = []
        for i, obs_cfg in enumerate(obstacle_cfgs):
            obs = RigidObject(obs_cfg)
            self.obstacles.append(obs)
            self.scene.rigid_objects[f"obstacle_{i}"] = obs

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
        
        # Handle potential NaNs or Infs from the policy to prevent CUDA TDR crashes in PhysX
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print(f"[ENV WARNING] NaN/Inf detected in actions! Policy weights have likely collapsed. Zeroing actions to prevent PhysX crash.")
            #actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.last_action.copy_(actions)
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[DEBUG FATAL] CUDA Error BEFORE _apply_action (likely from _pre_physics_step): {e}")
            raise
        
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

        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[DEBUG FATAL] CUDA Error AFTER _apply_action tensor ops: {e}")
            raise

    def _update_zoom(self, zoom_cmd: torch.Tensor):
        delta_zoom = zoom_cmd * self.cfg.robot_phys_cfg.zoom_speed
        self.current_focal_lengths += delta_zoom

        self.current_focal_lengths = torch.clamp(
            self.current_focal_lengths,
            self.cfg.robot_phys_cfg.min_focal_length,
            self.cfg.robot_phys_cfg.max_focal_length
        )
        
        # Only update RayCaster and USD if focal length changed by > 0.05
        diff = torch.abs(self.current_focal_lengths - self.committed_focal_lengths)
        update_mask = diff > 0.05
        
        if not torch.any(update_mask):
            return
            
        update_indices = update_mask.nonzero(as_tuple=False).squeeze(-1)
        
        focal_lengths_cpu_to_update = self.current_focal_lengths[update_indices].cpu().numpy()
        for idx_idx, env_idx in enumerate(update_indices.tolist()):
            self.camera_prims[env_idx].GetFocalLengthAttr().Set(float(focal_lengths_cpu_to_update[idx_idx]))
            if hasattr(self, "high_res_camera_prims") and len(self.high_res_camera_prims) > env_idx:
                self.high_res_camera_prims[env_idx].GetFocalLengthAttr().Set(float(focal_lengths_cpu_to_update[idx_idx]))

        self.committed_focal_lengths[update_mask] = self.current_focal_lengths[update_mask].clone()

        self._zoom_ray_caster(update_indices)
        
        if hasattr(self, "_ptz_camera"):
            self._ptz_camera._update_intrinsic_matrices(update_indices)
        if hasattr(self, "_high_res_ptz_camera"):
            self._high_res_ptz_camera._update_intrinsic_matrices(update_indices)
        
    def _zoom_ray_caster(self, env_ids=None):
        if env_ids is None:
            f = self.committed_focal_lengths.to(self.device)
            num_updates = self.num_envs
        else:
            f = self.committed_focal_lengths[env_ids].to(self.device)
            num_updates = len(env_ids)
            
        pcfg = self._raycaster_camera.cfg.pattern_cfg
        w, h = pcfg.width, pcfg.height
        ha = pcfg.horizontal_aperture
        va = pcfg.vertical_aperture if pcfg.vertical_aperture is not None else ha * (h / w)

        fx = w * f / ha
        fy = h * f / va
        cx = pcfg.horizontal_aperture_offset * fx + (w / 2.0)
        cy = pcfg.vertical_aperture_offset * fy + (h / 2.0)
        
        K = torch.zeros((num_updates, 3, 3), device=self.device, dtype=torch.float32)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        self._raycaster_camera.set_intrinsic_matrices(K, env_ids=env_ids) 

    def _update_maps(self, visualise: bool = False):
        
        if not self.cfg.mapping_cfg.use_occupancy_map:
            return
        
        chassis_min_dist_sq = 0.4 ** 2
        # Update Visitation Map
        self.map_manager.update_visitation(self.robot.data.root_pos_w)

        # Update Occupancy Map
        # ---------------------------------------------------------
        # NAVIGATION CAMERA (Occupancy Map)
        # ---------------------------------------------------------

        nav_cam_pos = self._nav_camera.data.pos_w
        nav_cam_quat = self._nav_camera.data.quat_w_ros

        nav_depth_data = self._nav_camera.data.output["distance_to_image_plane"]
        intrinsic_matrices = self._nav_camera.data.intrinsic_matrices   
        # Batched unprojection for navigation camera
        pointclouds_batch = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_matrices,
                depth=nav_depth_data,
                keep_invalid=True,  # Ensure we return full shape (N, P, 3)
                position=nav_cam_pos,
                orientation=nav_cam_quat,
                device=self.device,
        )
           # To visualise the point cloud in my Scene
        # if i ==0 and run_cfg.visualise_point_cloud and visualise and self.pc_markers is not None:
        #     if pointcloud.size()[0] > 0:
        #         self.pc_markers.visualize(translations=pointcloud)
        point_clouds_list  = []
        for i in range(self.num_envs):
            pointcloud = pointclouds_batch[i]
            
            # Remove NaN/Inf natively since keep_invalid=True left them in
            valid_depth_mask = torch.logical_and(~torch.isnan(pointcloud[:, 2]), ~torch.isinf(pointcloud[:, 2]))
            pointcloud = pointcloud[valid_depth_mask]

            if pointcloud.shape[0] > 0:
                # 1. Find the indices of all points that are not part of the floor
                if getattr(self.cfg.mapping_cfg, "filter_floor_occupancy", True):
                    floor_mask = pointcloud[:, 2] > 0.05
                else:
                    floor_mask = pointcloud[:, 2] > -10.0 # Don't filter
                
                # Filter out chassis points (prevent the robot from mapping its own bumper at the origin)
                dist_sq = torch.sum((pointcloud - nav_cam_pos[i])**2, dim=1)
                chassis_mask = dist_sq > chassis_min_dist_sq
                
                # Combine masks
                valid_mask = torch.logical_and(floor_mask, chassis_mask)
                valid_indices = torch.where(valid_mask)[0]

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
            point_clouds_list.append(pointcloud)
        
        self.map_manager.update_occupancy(
            sensor_origins=nav_cam_pos,
            point_clouds=point_clouds_list,
        )

        # ---------------------------------------------------------
        # INSPECTION CAMERA (Visibility Map)
        # ---------------------------------------------------------
        insp_cam_pos = self._ptz_camera.data.pos_w
        insp_cam_quat = self._ptz_camera.data.quat_w_ros

        depth_data_insp = self._ptz_camera.data.output["distance_to_image_plane"]
        intrinsic_matrices_insp = self._ptz_camera.data.intrinsic_matrices

        # Batched unprojection for inspection camera
        pointclouds_insp_batch = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_matrices_insp,
                depth=depth_data_insp,
                keep_invalid=True,  # Ensure we return full shape (N, P, 3)
                position=insp_cam_pos,
                orientation=insp_cam_quat,
                device=self.device,
        )

        point_clouds_list_insp = []        
        for i in range(self.num_envs):
            pointcloud_insp = pointclouds_insp_batch[i]
            
            # Remove NaN/Inf natively since keep_invalid=True left them in
            valid_depth_mask = torch.logical_and(~torch.isnan(pointcloud_insp[:, 2]), ~torch.isinf(pointcloud_insp[:, 2]))
            pointcloud_insp = pointcloud_insp[valid_depth_mask]

            if pointcloud_insp.shape[0] > 0:
                # Filter out chassis points
                dist_sq_insp = torch.sum((pointcloud_insp - insp_cam_pos[i])**2, dim=1)
                valid_mask_insp = dist_sq_insp > chassis_min_dist_sq
                pointcloud_insp = pointcloud_insp[valid_mask_insp]
            if pointcloud_insp.shape[0] > 1024:
                 perm = torch.randperm(pointcloud_insp.shape[0], device=self.device)
                 pointcloud_insp = pointcloud_insp[perm[:1024]]
            point_clouds_list_insp.append(pointcloud_insp)
        

        self.map_manager.update_visibility(
            sensor_origins=insp_cam_pos,
            point_clouds=point_clouds_list_insp,
        )

        # Update Visualizer
        if self.map_manager.visualizer is not None:
            # Pass the full batched poses for potential local map extraction
            robot_pos_w = self.robot.data.root_pos_w
            robot_quat_w = self.robot.data.root_quat_w
            
            # Call the updated visualization method with the pose data
            self.map_manager.update_visualization(robot_pos_w, robot_quat_w)
  
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
        robot_quat_w = self.robot.data.root_quat_w
        
        if getattr(self.cfg.mapping_cfg, "egocentric_map", True):
            robot_yaw_quat_w = yaw_quat(robot_quat_w)
        else:
            robot_yaw_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(robot_pos_w.shape[0], 1)
            
        local_occ_map, local_vis_map, local_visit = self.map_manager.get_local_maps(robot_pos_w, robot_yaw_quat_w)
        self.current_local_occ_map = local_occ_map.clone()
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

        nav_modality = getattr(run_cfg, "nav_camera_modality", "rgb")
        
        if nav_modality == "rgb":
            if "rgb" in self.cfg.sensor_cfg.navigation_camera.data_types:
                front_camera_data = self._nav_camera.data.output["rgb"] / 255.0
                if torch.isnan(front_camera_data).any():
                    print("[ENV DEBUG] NaN detected in Front Camera Data!")

        elif nav_modality == "depth":
             if "distance_to_image_plane" in self.cfg.sensor_cfg.navigation_camera.data_types:
                front_camera_data = self._nav_camera.data.output["distance_to_image_plane"].clone()
                # Replace Inf with a max range (e.g. 10.0m) to keep inputs sane
                front_camera_data[torch.isinf(front_camera_data)] = 10.0
                front_camera_data = front_camera_data.unsqueeze(-1)
                if torch.isnan(front_camera_data).any():
                    print("[ENV DEBUG] NaN detected in Front Camera Depth Data!")

        elif nav_modality == "rgbd":
            if "rgb" in self.cfg.sensor_cfg.navigation_camera.data_types and "distance_to_image_plane" in self.cfg.sensor_cfg.navigation_camera.data_types:
                rgb = self._nav_camera.data.output["rgb"] / 255.0
                depth = self._nav_camera.data.output["distance_to_image_plane"].clone()
                depth[torch.isinf(depth)] = 10.0
                if depth.dim() == 3:
                     depth = depth.unsqueeze(-1)
                
                front_camera_data = torch.cat([rgb, depth], dim=-1)
                if torch.isnan(front_camera_data).any():
                    print("[ENV DEBUG] NaN detected in Front Camera RGBD Data!")

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

        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[DEBUG FATAL] CUDA Error BEFORE environment step: {e}")
            raise

        if self.common_step_counter % self.cfg.mapping_cfg.map_update_interval == 0:
            try:
                self._update_maps(visualise=run_cfg.debug)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[DEBUG FATAL] CUDA Error inside `_update_maps` (Warp Kernels)! This proves the warp kernels are causing the timeout: {e}")
                raise

        self.logger.log_step(self.common_step_counter, self.curriculum.success_rate)

        if self.data_collector:
            self.data_collector.collect(self._ptz_camera, self.common_step_counter)

        if self.pc_collector:
             inspection_cam = getattr(self, "_high_res_ptz_camera", self._ptz_camera)
             target_mask = self._get_semantic_mask(inspection_cam)
             if target_mask is not None:
                 if hasattr(self, '_nav_camera'):
                     self.pc_collector.collect(inspection_cam, self.common_step_counter, semantic_mask=target_mask, nav_camera=self._nav_camera)
                 else:
                     self.pc_collector.collect(inspection_cam, self.common_step_counter, semantic_mask=target_mask)

        # Record max distance from origin taking into account X and Y
        current_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2], dim=-1)
        if hasattr(run_cfg, 'data_recording_path') and run_cfg.data_recording_path:
            self.max_distance_reached = torch.maximum(self.max_distance_reached, current_dist)

        try:
            res = super().step(action)
            torch.cuda.synchronize()
            return res
        except Exception as e:
            print(f"[DEBUG FATAL] CUDA Error inside `super().step()` (PhysX / Apply Action)! {e}")
            raise
    def _get_semantic_mask(self, camera) -> torch.Tensor | None:
        """
        Utility to extract the binary mask for the target object from a given camera.
        Returns a boolean tensor (N, H, W, 1) or None if target not found.
        """
        # Ensure data exists
        seg_data = camera.data.output.get("semantic_segmentation")
        if seg_data is None:
            return None
            
        info = camera.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        
        mask = torch.zeros_like(seg_data, dtype=torch.bool)
        
        # 1. Pre-compute mappings in Python (runs very fast since num unique classes is small)
        class_to_ids = {}
        for k, v in id_to_labels.items():
            cls_name = v.get("class")
            if cls_name not in class_to_ids:
                class_to_ids[cls_name] = []
            class_to_ids[cls_name].append(int(k))

        target_to_envs = {}
        for env_idx, target_name in enumerate(self.env_target_names):
            if target_name not in target_to_envs:
                target_to_envs[target_name] = []
            target_to_envs[target_name].append(env_idx)

        # 2. Vectorized mask application per target class
        for target_name, env_indices in target_to_envs.items():
            target_ids = class_to_ids.get(target_name, [])
            if not target_ids:
                continue
                
            env_indices_tensor = torch.tensor(env_indices, device=seg_data.device, dtype=torch.long)
            target_ids_tensor = torch.tensor(target_ids, device=seg_data.device, dtype=seg_data.dtype)
            
            # Using torch.isin applies the mask for all corresponding environments and IDs at once
            mask[env_indices_tensor] = torch.isin(seg_data[env_indices_tensor], target_ids_tensor)
                
        return mask if mask.any() else None
    
    def _update_max_ray_counts(self, valid_faces_tensor):
        """
        Helper: Checks if the current view of any face has more rays (better zoom) 
        than previously recorded.
        """
        # Count how many rays are hitting each unique face in this specific frame
        unique_ids, counts = torch.unique(valid_faces_tensor, return_counts=True)

        # Ensure we don't index out of bounds
        valid_mask = unique_ids < self.q_capacity
        unique_ids = unique_ids[valid_mask]
        counts = counts[valid_mask]

        if unique_ids.numel() == 0:
            return

        # Update maximums on the GPU
        current_maxes = self.face_max_counts[unique_ids]
        new_maxes = torch.maximum(current_maxes, counts)
        self.face_max_counts[unique_ids] = new_maxes
        
        # Update global max and only sync to CPU to print if a new global max is found (rare)
        batch_max = counts.max()
        if batch_max > self.global_max_ray_count[0]:
            self.global_max_ray_count[0] = batch_max
            max_idx = counts.argmax()
            print(f"!!! New Global Max Zoom Record: {batch_max.item()} rays (Face {unique_ids[max_idx].item()}) !!!")


    def _compute_face_discovery_reward_fast(self):
        """
        Compute the reward for discovering new faces.
        """
        face_ids = self._raycaster_camera.data.output.get("face_ids")
        target_mask = self._get_semantic_mask(self._ptz_camera)
        if run_cfg.debug and run_cfg.visualise_face_ids:
            self._show_face_ids_(
                face_ids=face_ids,
                target_mask=target_mask,
                env_id=0,
                win="face_ids_debug",
                scale=10,
                max_ids_in_text=12,
            )

         # Exit if either camera data is missing
        if target_mask is None or face_ids is None:
            return (torch.zeros(self.num_envs, device=self.device), 
                    torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
    
    
        occlusion_filtered_face_ids = torch.full_like(face_ids, -1)
        occlusion_filtered_face_ids[target_mask] = face_ids[target_mask]

        # per environments operation
        face_rewards = torch.zeros(self.num_envs, device=self.device)
        num_faces_inspected = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        k = self.cfg.reward_cfg.face_quality_k

        if self.cfg.reward_cfg.use_angle_weighted_reward:
            # Get normals: (Num_Envs, H, W, 3)
            normals = self._raycaster_camera.data.output["normals"]
            
            # Get ray directions in world frame: (Num_Envs, Num_Rays, 3)
            # We need to reshape to (Num_Envs, H, W, 3) to match normals
            # Note: Checking MultiMeshRayCasterCamera, num_rays = width * height
            H, W = normals.shape[1], normals.shape[2]
            ray_dirs = self._raycaster_camera._ray_directions_w.view(self.num_envs, H, W, 3)

            # View direction is roughly -ray_direction (vector from surface to camera)
            # Dot product: (N . V) = (N . -R) = -(N . R)
            # We want max(0, N . V)
            dot_prods = -torch.sum(normals * ray_dirs, dim=-1)
            weights = torch.clamp(dot_prods, min=0.0, max=1.0)
        else:
            # Fallback to pure counts (weight = 1.0)
            weights = torch.ones_like(face_ids, dtype=torch.float32).squeeze(-1)

        optical_flow_means = []
        optical_flow_excess_means = []
        optical_flow_multiplier_means = []
        
        if "motion_vectors" in self._ptz_camera.data.output:
            motion_vecs = self._ptz_camera.data.output["motion_vectors"]
            flow_magnitude = torch.norm(motion_vecs.float(), dim=-1)
            
            safe_zone = self.cfg.robot_phys_cfg.flow_safe_zone
            drop_speed = self.cfg.robot_phys_cfg.flow_drop_speed
            
            active_penalty = torch.clamp(flow_magnitude - safe_zone, min=0.0)
            flow_multiplier = torch.exp(-0.5 * torch.square(active_penalty / drop_speed))
            
            if getattr(run_cfg, "use_optical_flow_as_quality", False):
                weights = weights * flow_multiplier.squeeze(-1) if flow_multiplier.dim() > weights.dim() else weights * flow_multiplier

        # Apply Distance Mask to Weights
        if getattr(run_cfg, "use_depth_mask", False):
            depth = self._raycaster_camera.data.output["distance_to_image_plane"].squeeze(-1)
            max_dist = self.cfg.reward_cfg.max_inspection_distance
            sigma = self.cfg.reward_cfg.depth_sigma
            
            depth_mask = torch.where(
                depth <= max_dist,
                torch.ones_like(depth),
                torch.exp(-0.5 * torch.square((depth - max_dist) / sigma))
            )

            weights = weights * depth_mask

        visible_faces_counts = []

        for env_idx in range(self.num_envs):
            # Flatten everything for this env
            env_faces = occlusion_filtered_face_ids[env_idx].flatten()
            env_weights = weights[env_idx].flatten()

            # Filter valid faces
            valid_mask = env_faces >= 0
            valid_faces = env_faces[valid_mask]
            valid_weights = env_weights[valid_mask]

            if "motion_vectors" in self._ptz_camera.data.output:
                env_flow = flow_magnitude[env_idx].flatten()
                env_excess = active_penalty[env_idx].flatten()
                env_flow_mult = flow_multiplier[env_idx].flatten()
                
                valid_flow = env_flow[valid_mask]
                valid_excess = env_excess[valid_mask]
                valid_flow_mult = env_flow_mult[valid_mask]
                
                if valid_flow.numel() > 0:
                    optical_flow_means.append(valid_flow.mean().item())
                    optical_flow_excess_means.append(valid_excess.mean().item())
                    optical_flow_multiplier_means.append(valid_flow_mult.mean().item())

            unique_ids_visible = torch.unique(valid_faces)
            visible_faces_counts.append(len(unique_ids_visible))

            if valid_faces.numel() == 0:
                num_faces_inspected[env_idx] = (self.best_q_per_face[env_idx] > 0).sum()
                continue

            if run_cfg.debug and run_cfg.display_ray_counts:
                self._update_max_ray_counts(valid_faces)

            # We need sum of weights per unique face ID
            # Since IDs are integers, we can use scatter_add or bincount if memory allows, 
            # or loop through unique IDs if sparse.
            # Given q_capacity can be large but valid_faces in a frame is small (1024),
            # let's find unique IDs and sum weights for them.
            
            # Method:
            # 1. Find unique faces and their inverse indices to map pixels to unique faces
            unique_ids, inverse_indices = torch.unique(valid_faces, return_inverse=True)
            
            # 2. Sum weights for each unique face
            # Initialize with zeros
            weighted_counts = torch.zeros_like(unique_ids, dtype=torch.float32)
            # Add weights: index=inverse_indices adds valid_weights to weighted_counts
            weighted_counts.scatter_add_(0, inverse_indices, valid_weights)

            # Capacity ensures we don't go out of bounds. The max_id is safely accommodated.
            if unique_ids.max() >= self.q_capacity:
                print(f"[WARNING] Face ID {unique_ids.max()} exceeds capacity {self.q_capacity}! Ignoring out of bounds faces.")
                valid_bound = unique_ids < self.q_capacity
                unique_ids = unique_ids[valid_bound]
                weighted_counts = weighted_counts[valid_bound]

            # Calculate Quality: Q = 1 - exp(-WeightedCount / k)
            q_now = 1.0 - torch.exp(-weighted_counts / k)

            q_best = self.best_q_per_face[env_idx, unique_ids]

            dq = torch.relu(q_now - q_best)
            face_rewards[env_idx] = dq.sum()

            # Update best q values
            self.best_q_per_face[env_idx, unique_ids] = torch.maximum(q_best, q_now)
            num_faces_inspected[env_idx] = (self.best_q_per_face[env_idx] > 0).sum()
            
            # Normalize face reward by the total number of mesh faces
            face_rewards[env_idx] /= self.total_mesh_faces[env_idx].float()
        
        self.logger.log_visible_faces(np.mean(visible_faces_counts))
        if len(optical_flow_means) > 0:
            self.logger.log_optical_flow(np.mean(optical_flow_means), np.mean(optical_flow_excess_means), np.mean(optical_flow_multiplier_means))
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
    
    _show_face_ids_ = _show_face_ids_
    
    def _compute_exploration_rewards(self):
        # --- Entropy / Information Gain ---
        all_maps_log_odds = wp.to_torch(self.map_manager.occupancy_map)
        all_maps_log_odds = all_maps_log_odds.view(self.num_envs, -1)
        current_entropy = self._calculate_entropy(all_maps_log_odds).sum(dim=1)
        information_gain = torch.relu(self.prev_global_map_entropy - current_entropy)
        self.prev_global_map_entropy = current_entropy.clone()

        # ---Surface Visibility Increase ---
        k = self.cfg.reward_cfg.visibility_decay_factor
        all_vis_maps = wp.to_torch(self.map_manager.visibility_map)
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
        map_origins = torch.from_numpy(self.map_manager.world_map_origins).to(self.device)
        voxel_size = self.map_manager.resolution
        map_dims = self.map_manager.map_dims

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
        map_offset = env_ids * self.map_manager.num_voxels_per_map
        global_indices = map_offset + linear_indices

        # Get visitation counts from the GPU map
        all_visit_counts_torch = wp.to_torch(self.map_manager.visitation_map)
        current_counts = all_visit_counts_torch[global_indices]

        # Calculate reward: R = exp(-beta * N), where N is the visit count
        # Note: We subtract 1.0 because the map was already updated this step.
        # We want to reward based on the state *before* the current visit.
        reward[valid_mask] = torch.exp(-self.cfg.reward_cfg.visitation_decay_factor * (current_counts - 1.0))
        
        return reward

    def _compute_occupancy_penalty(self) -> torch.Tensor:
        """
        Computes a collision penalty based on occupancy map voxels directly surrounding the robot.
        Uses a shift window from the egocentric map center.
        """
        if getattr(self, 'current_local_occ_map', None) is None:
            return torch.zeros(self.num_envs, device=self.device)
            
        local_dims = self.cfg.mapping_cfg.local_map_dims
        center_x = local_dims[0] // 2
        center_y = local_dims[1] // 2
        center_z = 0 # Robot is at the floor level in the local map
        
        import math
        # Default target collision check radius is 0.5 meters. Calculate voxels dynamically based on resolution.
        target_radius_m = 0.5
        calculated_shift = max(1, int(math.ceil(target_radius_m / self.cfg.mapping_cfg.resolution)))
        
        # Allow overriding via config, otherwise use the dynamically calculated shift
        shift_x = getattr(self.cfg.reward_cfg, 'occupancy_penalty_shift_x', calculated_shift)
        shift_y = getattr(self.cfg.reward_cfg, 'occupancy_penalty_shift_y', calculated_shift)
        shift_z = getattr(self.cfg.reward_cfg, 'occupancy_penalty_shift_z', calculated_shift)

        # Ensure bounds
        min_x = max(0, center_x - shift_x)
        max_x = min(local_dims[0], center_x + shift_x + 1)
        min_y = max(0, center_y - shift_y)
        max_y = min(local_dims[1], center_y + shift_y + 1)
        min_z = max(0, center_z) # Can't go below floor
        max_z = min(local_dims[2], center_z + shift_z + 1)

        # Log-odds of 1.1 corresponds to a high probability of occupancy
        collision_mask = self.current_local_occ_map[..., min_x:max_x, min_y:max_y, min_z:max_z] > 1.1
        
        collision_mask_size = collision_mask.shape[1] * collision_mask.shape[2] * collision_mask.shape[3]
        occupancy_penalty_sum = collision_mask.sum(dim=(1,2,3))
        
        occupancy_penalty = torch.where(
            occupancy_penalty_sum > 0.0, 
            -(1.0 + occupancy_penalty_sum / collision_mask_size), 
            0.0
        )
        return occupancy_penalty

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
        occupancy_penalty = self._compute_occupancy_penalty()
        action_delta = torch.sum(torch.square(self.actions - self.previous_action_for_rewards), dim=1)
        
        # Base Action Penalty (Linear & Angular velocity which are indices 0, 1)
        base_action_delta = torch.sum(torch.square(
            self.actions[:, :2] - self.previous_action_for_rewards[:, :2]
        ), dim=1)

        # Camera Penalty (Pan, Tilt, Zoom which are indices 2, 3, 4)
        ptz_action_delta = torch.sum(torch.square(
            self.actions[:, 2:] - self.previous_action_for_rewards[:, 2:]
        ), dim=1)

        # Inpsection Coverage Ratio and Success Bonus
        current_coverage_ratio = total_num_faces_inspected / self.total_mesh_faces
        self.prev_coverage_ratio = current_coverage_ratio.clone()
       
        success_bonus = torch.where(
            current_coverage_ratio >= self.curriculum.get_current_coverage_goal(), 
            self.cfg.reward_cfg.coverage_reward,
            0.0
        )

        # --- Adaptive Reward Scaling ---
        # Decay the face discovery reward as curriculum progresses
        # We want it high initially (to learn what faces are) and lower later (to prioritize exploration)
        # progress = self.curriculum.get_progress()
        
        # # Linear Decay: Scales from 1.0 down to 0.2 (20% of original)
        # decay_factor = 1.0 - (progress * 0.8) 
        # current_face_reward_scale = self.cfg.reward_cfg.mesh_coverage_reward_scale * decay_factor
        current_face_reward_scale = self.cfg.reward_cfg.mesh_coverage_reward_scale

        total_reward = (current_face_reward_scale * face_discovery_raw
                        + self.cfg.reward_cfg.information_gain_reward_scale * information_gain_reward
                        + self.cfg.reward_cfg.visitation_reward_scale * visitation_reward # Added visitation reward
                        - self.cfg.reward_cfg.action_penalty_scale * base_action_delta
                        - self.cfg.reward_cfg.ptz_penalty_scale * ptz_action_delta
                        + getattr(self.cfg.reward_cfg, 'occupancy_penalty_scale', 1.0) * occupancy_penalty
                        + success_bonus
                        -self.cfg.reward_cfg.time_penalty
                        )
        self._cache_rewards(
            face_discovery_raw,
            information_gain_reward, 
            visibility_increase_reward, 
            base_action_delta,
            ptz_action_delta,
            visitation_reward,
            occupancy_penalty,
            total_reward,
            current_face_reward_scale # Pass the dynamic scale for logging
            )
        # print(f"[DEBUG] Total Reward before scaling: {total_reward}")
        # Logging
      
        if torch.isnan(total_reward).any() or torch.isinf(total_reward).any():
            print("NaN or Inf detected in total reward calculation!")
            print(f"Total Reward: {total_reward}")
            print(f"face_discovery_raw: {face_discovery_raw}")
            print(f"information_gain_reward: {information_gain_reward}")
            print(f"visibility_increase_reward: {visibility_increase_reward}")
            print(f"visitation_reward: {visitation_reward}")
            print(f"base_action_delta: {base_action_delta}")
            print(f"ptz_action_delta: {ptz_action_delta}")
            print(f"occupancy_penalty: {occupancy_penalty}")
            print(f"success_bonus: {success_bonus}")
            raise ValueError("Training stopped: NaN or Inf detected in total reward calculation!")
        # return total_reward
        normalized_reward = self.rewardscaler(total_reward)
        return normalized_reward
    
    def _cache_rewards(self, face_discovery, info_gain, visibility_increase, action_delta, camera_delta, visitation_reward, occupancy_penalty, total_unscaled, current_face_reward_scale):
        # Construct dictionary for logging (both accumulation and per-step means)
        reward_dict = {
            "face_discovery_raw": face_discovery,
            "info_gain": info_gain,
            "visibility_increase": visibility_increase,
            "action_penalty": action_delta,
            "camera_penalty": camera_delta,
            "visitation_reward": visitation_reward,
            "occupancy_penalty": occupancy_penalty,
            "total_unscaled": total_unscaled,
            
            # Scaled
            "face_discovery_scaled": current_face_reward_scale * face_discovery,
            "info_gain_scaled": self.cfg.reward_cfg.information_gain_reward_scale * info_gain,
            "visibility_increase_scaled": self.cfg.reward_cfg.visibility_increase_reward_scale * visibility_increase,
            "action_penalty_scaled": self.cfg.reward_cfg.action_penalty_scale * action_delta,
            "camera_penalty_scaled": self.cfg.reward_cfg.ptz_penalty_scale * camera_delta,
            "visitation_reward_scaled": self.cfg.reward_cfg.visitation_reward_scale * visitation_reward,
            "occupancy_penalty_scaled": getattr(self.cfg.reward_cfg, 'occupancy_penalty_scale', 1.0) * occupancy_penalty,
            
            # Debug Stats - Must be a tensor of shape (num_envs,) for the logger to handle it correctly during resets
            "stats/face_reward_scale": torch.full((self.num_envs,), current_face_reward_scale, device=self.device)
        }
        
        self.logger.accumulate_rewards(reward_dict)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for timeout
        # Check for timeout
        max_steps = self.curriculum.get_current_episode_length()
        time_out = self.episode_length_buf >= max_steps - 1
        current_q_values = self.best_q_per_face
        num_faces_inspected = (current_q_values > 0.0).sum(dim=1)

        total_quality = current_q_values.sum(dim=1)
        mean_quality = torch.zeros_like(total_quality)
        mask = num_faces_inspected > 0
        mean_quality[mask] = total_quality[mask] / num_faces_inspected[mask].float()
        coverage_achieved = (num_faces_inspected / self.total_mesh_faces) >= self.curriculum.get_current_coverage_goal()
        success_condition = coverage_achieved 
        return success_condition, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Print reset info for debugging
        if run_cfg.debug and ((isinstance(env_ids, torch.Tensor) and 0 in env_ids) or (isinstance(env_ids, list) and 0 in env_ids)):
             print(f"[DEBUG] Env 0 RESETTING at Step: {self.common_step_counter}. Time: {self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation:.2f}s")


        if len(env_ids)> 0:
            self.last_action[env_ids] = 0.0
            self.previous_action_for_rewards[env_ids] = 0.0


            current_q_values = self.best_q_per_face[env_ids]
            num_faces_inspected = (current_q_values > 0.0).sum(dim=1)
            # Add to extras for external logging
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["faces_discovered"] = num_faces_inspected
            if hasattr(run_cfg, 'data_recording_path') and run_cfg.data_recording_path:
                self.extras["log"]["max_distance"] = self.max_distance_reached[env_ids]
            
            total_quality = current_q_values.sum(dim=1)

            mean_quality = torch.zeros_like(total_quality)
            mask = num_faces_inspected > 0
            mean_quality[mask] = total_quality[mask] / num_faces_inspected[mask].float()

            achieved_coverage_ratios = num_faces_inspected / self.total_mesh_faces[env_ids].float()
            current_cov_goal = self.curriculum.get_current_coverage_goal()
            episode_successes = (achieved_coverage_ratios >= current_cov_goal) # & (mean_quality >= self.curriculum.get_current_quality_goal())
            self.curriculum.update_curriculum(episode_successes, mean_quality)

            # Check if curriculum updated
            new_cov_goal = self.curriculum.get_current_coverage_goal()
            if new_cov_goal != current_cov_goal:
                self.logger.clear_episode_buffers()


            num_active_obstacles = self.curriculum.get_num_active_obstacles(self.cfg.max_obstacles)

            # Logging
            for i, env_id in enumerate(env_ids):
                if env_id == 0:
                    final_face_count = num_faces_inspected[i].item()
                    if hasattr(run_cfg, 'data_recording_path') and run_cfg.data_recording_path:
                        max_dist = self.max_distance_reached[i].item()
                        if hasattr(self, 'pc_collector') and self.pc_collector is not None:
                            self.pc_collector.flush_if_best(final_face_count, max_dist)
                            
                    if getattr(run_cfg, 'debug', False):
                        print(f"--- Episode Summary Env 0 --- Final Faces Discovered: {final_face_count} ---")

                self.episode_log_buffer["coverage_percent"].append(achieved_coverage_ratios[i].item() * 100)
                self.episode_log_buffer["faces_discovered"].append(num_faces_inspected[i].item())
                self.logger.update_episode_stats(num_faces_inspected[i].item()) # Update global stats
                self.episode_log_buffer["mean_inspection_quality"].append(mean_quality[i].item())
                self.episode_log_buffer["curriculum/current_threshold"].append(current_cov_goal)
                self.episode_log_buffer["curriculum/active_obstacles"].append(num_active_obstacles)
                self.episode_log_buffer["curriculum/task_area"].append(self.curriculum.get_total_task_area())
                
            # Log and Reset Reward Sums
            self.logger.log_and_reset_episode_rewards(env_ids)

            # Logging Loop Continue
            for i, env_id in enumerate(env_ids):
                # Clear Buffers
                
                self.prev_coverage_ratio[env_id] = 0.0

            self.best_q_per_face[env_ids] = 0.0
                
            # Map Logging and Reset
            if self.cfg.mapping_cfg.use_occupancy_map:

                all_vis_maps_torch = wp.to_torch(self.map_manager.visibility_map).view(self.num_envs, -1)
                all_occ_maps_torch = wp.to_torch(self.map_manager.occupancy_map).view(self.num_envs, -1)
                all_visit_maps_torch = wp.to_torch(self.map_manager.visitation_map).view(self.num_envs, -1) # NEW

                final_vis_sums = all_vis_maps_torch.sum(dim=1)
                final_entropies = self._calculate_entropy(all_occ_maps_torch).sum(dim=1)
                final_robot_path_cells = (all_visit_maps_torch > 0).sum(dim=1)
                final_unique_visible_cells = (all_vis_maps_torch > 0).sum(dim=1)

                for env_id in env_ids.cpu().tolist():
                    # Covergae for Trajectory
                    self.episode_log_buffer["final_visited_cells_count"].append(final_robot_path_cells[env_id].item())
                    #Surface Coverage Metric
                    self.episode_log_buffer["final_unique_visible_cell_count"].append(final_unique_visible_cells[env_id].item())
                    # Map Entropy
                    self.episode_log_buffer["final_map_entropy"].append(final_entropies[env_id].item())
                    
                    # Entropy % against Max Theoretical Entropy
                    # max entropy per cell = 1.0. Therefore max theoretically possible is just num_cells.
                    num_total_cells = all_occ_maps_torch.shape[1]
                    entropy_percent = (final_entropies[env_id].item() / num_total_cells) * 100.0
                    self.episode_log_buffer["episode_summary/final_map_entropy_percent"].append(entropy_percent)
                    if run_cfg.debug and env_id == 0:
                        print(f"--- Episode Summary Env 0 --- Final Unique Visible Cells: {final_unique_visible_cells[env_id].item()} ---")
                        print(f"--- Episode Summary Env 0 --- Final Map Entropy: {final_entropies[env_id].item()} ---")

                self.map_manager.reset_map(env_ids.cpu().tolist())
                all_vis_maps = wp.to_torch(self.map_manager.visibility_map)
                num_cells = all_vis_maps.view(self.num_envs, -1).shape[1]
                initial_score = -float(num_cells)
                self.prev_visibility_score[env_ids] = initial_score 
                all_maps_log_odds = wp.to_torch(self.map_manager.occupancy_map)
                all_maps_log_odds = all_maps_log_odds.view(self.num_envs, -1)
                initial_entropy = self._calculate_entropy(all_maps_log_odds).sum(dim=1)
                self.prev_global_map_entropy[env_ids] = initial_entropy[env_ids]
                
                self.prev_local_occ_map[env_ids] = 0.0
                self.prev_local_vis_map[env_ids] = 0.0
                if hasattr(run_cfg, 'data_recording_path') and run_cfg.data_recording_path:
                    self.max_distance_reached[env_ids] = 0.0


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
        
        # --- Update Information Object Position ---
        # Get new objective positions based on curriculum and robot position
        # We pass the robot's *local* position (new_pos) to the curriculum manager
        obj_pos, obj_quat = self.curriculum.get_objective_start_pos(num_resets, new_pos[:, :3])
        
        obj_state = torch.zeros((num_resets, 13), device=self.device)
        obj_state[:, :3] = obj_pos + self.scene.env_origins[env_ids]
        obj_state[:, 3:7] = obj_quat

        env_ids_local = env_ids.cpu().numpy()

        for target_name, target_cfg in self.cfg.inspection_goal_cfg.inspection_targets.items():
            # Boolean mask for environments assigned this target
            assigned_mask_np = (self.env_target_names[env_ids_local] == target_name)
            assigned_mask = torch.from_numpy(assigned_mask_np).to(device=self.device, dtype=torch.bool)
            
            mixed_state = torch.zeros((num_resets, 13), device=self.device)
            hidden_pos = torch.tensor([0, 0, -100.0], device=self.device)
            mixed_state[:, :3] = hidden_pos + self.scene.env_origins[env_ids]
            mixed_state[:, 3] = 1.0  # Identity quat w
            
            if assigned_mask.any():
                mixed_state[assigned_mask, :3] = obj_state[assigned_mask, :3]
                target_ori = torch.tensor(target_cfg.orientation, device=self.device, dtype=torch.float32)
                mixed_state[assigned_mask, 3:7] = target_ori
                mixed_state[assigned_mask, 7:] = obj_state[assigned_mask, 7:]
            
            self.inspection_goals[target_name].write_root_pose_to_sim(mixed_state[:, :7], env_ids)
            self.inspection_goals[target_name].write_root_velocity_to_sim(mixed_state[:, 7:], env_ids)

            mixed_pos_cpu = mixed_state[:, :3].cpu().numpy().astype(float)
            mixed_ori_cpu = mixed_state[:, 3:7].cpu().numpy().astype(float)
            for i, env_id in enumerate(env_ids_local):
                xf = self.goal_prims_dict[target_name][env_id]
                if xf:
                    found_translate = False
                    found_orient = False
                    for op in xf.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            try:
                                op.Set(Gf.Vec3d(*mixed_pos_cpu[i]))
                            except Exception:
                                op.Set(Gf.Vec3f(*mixed_pos_cpu[i]))
                            found_translate = True
                        elif op.GetOpType() in [UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeTransform]:
                            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                q_w, q_x, q_y, q_z = float(mixed_ori_cpu[i, 0]), float(mixed_ori_cpu[i, 1]), float(mixed_ori_cpu[i, 2]), float(mixed_ori_cpu[i, 3])
                                try:
                                    op.Set(Gf.Quatd(q_w, q_x, q_y, q_z))
                                except Exception:
                                    op.Set(Gf.Quatf(q_w, q_x, q_y, q_z))
                                found_orient = True
                                
                    if not found_translate:
                        xf.AddTranslateOp().Set(Gf.Vec3d(*mixed_pos_cpu[i]))
                    if not found_orient:
                        xf.AddOrientOp().Set(Gf.Quatd(float(mixed_ori_cpu[i, 0]), float(mixed_ori_cpu[i, 1]), float(mixed_ori_cpu[i, 2]), float(mixed_ori_cpu[i, 3])))
        # ------------------------------------------

        # --- Base Spawning Logic ---
        existing_positions = [new_pos]
        existing_positions.append(obj_pos)

        # Spawn obstacles dynamically
        num_active_obstacles = self.curriculum.get_num_active_obstacles(self.cfg.max_obstacles)
        
        for i, obstacle in enumerate(self.obstacles):
            obstacle_state = torch.zeros((num_resets, 13), device=self.device)
            if i < num_active_obstacles:
                obs_pos, obs_quat = self.curriculum.get_obstacle_start_pos(num_resets, existing_positions)
                obstacle_state[:, :3] = obs_pos + self.scene.env_origins[env_ids]
                obstacle_state[:, 3:7] = obs_quat
                existing_positions.append(obs_pos)
            else:
                # Place inactive obstacles far below the ground plane
                hidden_pos = torch.zeros((num_resets, 3), device=self.device)
                hidden_pos[:, 2] = -100.0
                obstacle_state[:, :3] = hidden_pos + self.scene.env_origins[env_ids]
                obstacle_state[:, 3] = 1.0 # identity quat w=1
            
            obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(obstacle_state[:, 7:], env_ids)
        
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

    def _count_total_mesh_faces(self) -> int:
        """
        Performs a 'Pre-Scan' of the object to count observable faces.
        Teleports the raycaster camera to multiple positions around the object (in env 0)
        and accumulates the unique face IDs seen. 
        This is more accurate than raw mesh counts as it ignores internal/occluded faces.
        """
        print("[INFO] Starting Pre-Scan to count observable faces...")
        
        # 1. Setup Scan Views (Hemisphere around object)
        # Object position (from __init__ spawn)
        target_pos = torch.tensor([2.0, 0.0, 0.4], device=self.device)
        radius = 1.5
        
        # Generate view positions
        # Ring 1: Robot height approx
        azimuths = torch.linspace(0, 2*np.pi, 12, device=self.device)
        elevations = torch.tensor([0.2, 0.6], device=self.device) # Low and High angle
        
        view_positions = []
        for el in elevations:
            z = target_pos[2] + el
            # Radius shrinks for higher elevation to keep focusing on object? No, just cylinder/cone logic.
            # Simple cylinder rings
            for az in azimuths:
                x = target_pos[0] + radius * torch.cos(az)
                y = target_pos[1] + radius * torch.sin(az)
                view_positions.append(torch.tensor([x, y, z], device=self.device))
                
        view_positions = torch.stack(view_positions)
        
        # Compute orientations (look at target)
        # forward = target - eye
        forward = target_pos - view_positions
        forward = F.normalize(forward, dim=-1)
        
        # Assume up vector is Z world
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward)
        
        # Use simple look-at logic (Gram-Schmidt)
        # z_axis = -forward (Camera looks down -Z conventionally, but let's check config convention)
        # Config says convention="ros", so Forward is +Z? Or +X?
        # Standard ROS camera: X=Right, Y=Down, Z=Forward.
        # Isaac Lab Camera usually follows USD: -Z forward, +Y up.
        # Let's check quat_from_view_dirs usage in codebase or assume standard LookAt.
        # Better: use a helper or try generic lookat.
        
        # Re-using math utils usually safest. 
        # But wait, sensors_cfg says convention="ros". 
        # In ROS: Z is forward.
        
        # Simplified LookAt (Z forward):
        z_axis = forward
        x_axis = torch.linalg.cross(up, z_axis)
        x_axis = F.normalize(x_axis, dim=-1)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        
        # Rotation Matrix [X, Y, Z] columns
        rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        from isaaclab.utils.math import quat_from_matrix
        view_quats = quat_from_matrix(rot_mat)

        # 2. Iterate and Scan
        unique_faces_seen = set()
        
        # Save current state (though we are in init, so maybe not strictly needed, but good practice)
        # Note: We can't easily 'save' the camera state if it's attached, but we can just let it reset later.
        
        env_ids = torch.tensor([0], device=self.device)
        
        for i in range(len(view_positions)):
            pos = view_positions[i].unsqueeze(0) + self.scene.env_origins[0] # Add global offset
            quat = view_quats[i].unsqueeze(0)
            
            self._raycaster_camera.set_world_poses(pos, quat, env_ids)
            self._raycaster_camera.update(dt=0.0)
            
            # Get IDs
            face_ids = self._raycaster_camera.data.output["face_ids"][0].cpu().numpy().flatten()
            valid_ids = face_ids[face_ids >= 0]
            unique_faces_seen.update(valid_ids)
            
        total_observable = len(unique_faces_seen)
        
        if total_observable == 0:
            print("[WARNING] Pre-Scan found 0 faces! Falling back to raw mesh count or config.")
            return self.cfg.max_faces_to_inspect # Fallback
            
        print(f"[INFO] Pre-Scan Complete. Observable Faces: {total_observable} (Raw Mesh may be higher)")
        
        return total_observable