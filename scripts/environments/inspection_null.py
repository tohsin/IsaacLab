# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspection environment null test.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras =  True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.robot_inspection.utils.keyboard_controller import InspectionKeyboardController

"""Rest everything follows."""
import math
import gymnasium as gym
import torch
import cv2
import numpy as np
from isaaclab_tasks.direct.robot_inspection.run_config import cfg_mode as run_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = 42
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:

        # print info (this is vectorized environment)
        #cartpole
        #INFO]: Gym observation space: Box(-inf, inf, (1, 100, 100, 3), float32)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment

        env.reset()
        # Initialize the teleop keyboard controller
        keyboard_controller = InspectionKeyboardController(device=env.unwrapped.device, max_vel_speed = 0.8)
        print("[INFO]: Keyboard Controller Initialized.")
        print("[INFO]: Use Arrow Keys (UP/DOWN/LEFT/RIGHT) to move the robot base.")
        print("[INFO]: Use A/S/D/X to pan/tilt the PTZ camera (S: Up, X: Down, A: Left, D: Right).")

        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():

                for i in range(2500):
                    actions = keyboard_controller.advance()
                    
                    # Broadcast action if multiple environments are running
                    # if env.num_envs > 1:
                    #     actions = actions.repeat(env.num_envs, 1)
                        
                    obs, rewards, terminated, truncated, info  = env.step(actions)
                    obs_v = obs['policy']

                    if getattr(run_cfg, "display_cameras", False):
                        try:
                            # RGB image (1, H, W, 4) -> (H, W, 3)
                            ptz_rgb = env.unwrapped._ptz_camera.data.output["rgb"][0].cpu().numpy()
                            if ptz_rgb.shape[-1] == 4:
                                ptz_rgb = ptz_rgb[..., :3]
                                
                            if ptz_rgb.dtype != np.uint8:
                                if ptz_rgb.max() <= 1.0:
                                    ptz_rgb = (ptz_rgb * 255).astype(np.uint8)
                                else:
                                    ptz_rgb = ptz_rgb.astype(np.uint8)
                                    
                            # Semantic mask
                            ptz_semantic_raw = env.unwrapped._get_semantic_mask(env.unwrapped._ptz_camera)
                            num_observed_faces = 0
                            
                            if ptz_semantic_raw is not None:
                                try:
                                    face_ids = env.unwrapped._raycaster_camera.data.output.get("face_ids")
                                    if face_ids is not None:
                                        f_ids = face_ids[0].squeeze(-1).cpu().numpy()
                                        t_mask = ptz_semantic_raw[0].squeeze(-1).cpu().numpy()
                                        valid_mask = f_ids >= 0
                                        target_mask = t_mask > 0
                                        observed_faces = f_ids[valid_mask & target_mask]
                                        num_observed_faces = len(np.unique(observed_faces))
                                except Exception as e:
                                    pass

                                ptz_semantic = ptz_semantic_raw[0].cpu().numpy()
                                if ptz_semantic.ndim == 3 and ptz_semantic.shape[-1] == 1:
                                    ptz_semantic = ptz_semantic.squeeze(-1)
                                ptz_semantic_colored = cv2.applyColorMap((ptz_semantic * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            else:
                                ptz_semantic_colored = np.zeros_like(ptz_rgb)
                                
                            # Resize for better visibility
                            ptz_rgb = cv2.resize(ptz_rgb, (384, 384), interpolation=cv2.INTER_NEAREST)
                            ptz_semantic_colored = cv2.resize(ptz_semantic_colored, (384, 384), interpolation=cv2.INTER_NEAREST)
                            
                            # Isaac Sim outputs RGB, OpenCV expects BGR
                            ptz_bgr = cv2.cvtColor(ptz_rgb, cv2.COLOR_RGB2BGR)
                            
                            # Stack side by side
                            display_img = np.hstack((ptz_bgr, ptz_semantic_colored))
                            
                            text_x = 384 + 10
                            text_y = 30
                            cv2.putText(display_img, f"Live Faces: {num_observed_faces}", 
                                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            total_observed = 0
                            if hasattr(env.unwrapped, "best_q_per_face"):
                                total_observed = int((env.unwrapped.best_q_per_face[0] > 0).sum().item())
                                
                            cv2.putText(display_img, f"Total Faces: {total_observed}", 
                                        (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.imshow("PTZ Camera: RGB (Left) | Semantic (Right)", display_img)
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"[DEBUG] Camera display error: {e}")
                #now 
    finally:
          env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
