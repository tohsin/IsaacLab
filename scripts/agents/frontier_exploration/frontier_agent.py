# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a frontier exploration agent in the Isaac Lab environment."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Frontier Exploration agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import numpy as np
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Import our custom frontier logic
from frontier_utils import find_frontiers, cluster_frontiers, select_best_frontier

def get_action_towards_goal(robot_pose: torch.Tensor, goal_world: tuple) -> torch.Tensor:
    """
    Computes a simple proportional control action to drive the robot towards a goal.
    
    Args:
        robot_pose: [x, y, z, qx, qy, qz, qw] tensor or similar, containing position and orientation.
        goal_world: (x, y) tuple of the goal in world coordinates.
        
    Returns:
        Action tensor [v, omega, ptz_pan, ptz_tilt, zoom]
    """
    # Note: This is a placeholder for a real path planner (like A* or DWA)
    # This just blindly drives towards the goal.
    
    # Extract robot x, y, and yaw
    rx, ry = robot_pose[0].item(), robot_pose[1].item()
    
    # Simple conversion from quaternion to yaw (assuming standard format)
    # The exact indexing depends on your pose_obs format in inspection_env.py
    # Here we assume a simplified 2D planar movement for demonstration.
    
    gx, gy = goal_world
    
    dx = gx - rx
    dy = gy - ry
    
    target_yaw = math.atan2(dy, dx)
    
    # Note: You'll need to extract current_yaw from robot_pose
    current_yaw = 0.0 # TODO: Replace with actual yaw calculation from robot_pose quaternion
    
    yaw_error = target_yaw - current_yaw
    # Normalize yaw error to [-pi, pi]
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
    
    dist = math.sqrt(dx**2 + dy**2)
    
    # Control logic
    turn_speed = max(min(yaw_error * 1.5, 1.0), -1.0)
    
    if abs(yaw_error) > 0.5:
        # Turn in place if facing the wrong way
        fwd_speed = 0.0
    else:
        # Move forward while adjusting heading
        fwd_speed = max(min(dist, 1.0), 0.1)
        
    # Return [v, w, ptz_pan, ptz_tilt, zoom]
    return torch.tensor([[fwd_speed, turn_speed, 0.0, 0.0, 0.0]])


def main():
    """Frontier Exploration agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = 42
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        
        obs, _ = env.reset()
        
        while simulation_app.is_running():
            with torch.inference_mode():
                # For simplicity, we process only the first environment (index 0)
                # Ensure we are using the 'policy' dictionary
                if isinstance(obs, dict) and "policy" in obs:
                    obs_dict = obs["policy"]
                elif isinstance(obs, dict):
                    obs_dict = obs
                else:
                    raise ValueError("Expected a dictionary observation containing 'local-map' or 'global-map'")
                
                # Extract map from observations
                # Update this key if you expose 'global-map' later
                map_tensor = obs_dict.get('local-map', None)
                pose_tensor = obs_dict.get('robot-pose', None)
                
                if map_tensor is None or pose_tensor is None:
                    # Fallback to random actions if maps aren't correctly exposed
                    actions = torch.rand((env.num_envs, 5), device=env.unwrapped.device) * 2.0 - 1.0
                else:
                    # 1. Convert PyTorch tensor map to NumPy grid for OpenCV/SciPy processing
                    # Assuming map_tensor shape is [num_envs, height, width, channels]
                    # We take the occupancy channel (usually index 0) for env 0
                    occ_map_np = map_tensor[0, ..., 0].cpu().numpy()
                    
                    # 2. Find frontiers
                    frontiers = find_frontiers(occ_map_np, free_thresh=-0.1, unknown_thresh=0.1)
                    
                    # 3. Cluster frontiers
                    clusters = cluster_frontiers(frontiers)
                    
                    # 4. Select best frontier
                    # Assuming robot is always in the center of the local map for an egocentric view
                    map_h, map_w = occ_map_np.shape
                    robot_px = (map_h // 2, map_w // 2)
                    
                    best_frontier_px = select_best_frontier(clusters, robot_px)
                    
                    if best_frontier_px is not None:
                        # 5. Convert frontier pixel to world coordinates and plan path
                        # (This requires knowing map resolution and origin from SpatialStateManager)
                        # Placeholder: Map pixels to relative coordinates (e.g. 0.2m per pixel)
                        resolution = 0.2 
                        relative_y = (best_frontier_px[0] - robot_px[0]) * resolution
                        relative_x = (best_frontier_px[1] - robot_px[1]) * resolution
                        
                        # Note: Our get_action_towards_goal assumes world coordinates, 
                        # but if you pass relative coordinates, you can assume robot is at (0,0, yaw=0).
                        # Let's adjust the action generator inputs accordingly.
                        fake_robot_pose = torch.tensor([0.0, 0.0]) # local origin
                        goal_coord = (relative_x, relative_y)
                        
                        action_0 = get_action_towards_goal(fake_robot_pose, goal_coord)
                    else:
                        # No frontiers found (map fully explored or stuck)
                        # Spin to look around
                        action_0 = torch.tensor([[0.0, -1.0, 0.0, 0.0, 0.0]])
                        
                    # Apply action to all environments (or implement per-env logic)
                    actions = action_0.repeat(env.num_envs, 1).to(env.unwrapped.device)

                obs, rewards, terminated, truncated, info = env.step(actions)
                
                if terminated.any() or truncated.any():
                    print("[INFO] Episode terminated/truncated. Resetting...")
                    obs, _ = env.reset()
                    
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Closing environment...")
    finally:
        print("[INFO] Finalizing...")
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
