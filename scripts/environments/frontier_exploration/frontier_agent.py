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
from a_star import a_star

def get_action_towards_goal(robot_pose: torch.Tensor, goal_world: tuple) -> torch.Tensor:
    """
    Robot goes towards the goal.
    
    Args:
        robot_pose: [x, y, z, qx, qy, qz, qw] tensor or similar, containing position and orientation.
        goal_world: (x, y) tuple of the goal in world coordinates.
        
    Returns:
        Action tensor [v, omega, ptz_pan, ptz_tilt, zoom]
    """
    
    # Extract robot x, y, and yaw
    rx, ry = robot_pose[0].item(), robot_pose[1].item()
    
    # Simple conversion from quaternion to yaw (assuming standard format)
    # The exact indexing depends on your pose_obs format in inspection_env.py
    # Here we assume a simplified 2D planar movement for demonstration.
    
    gx, gy = goal_world
    
    dx = gx - rx
    dy = gy - ry
    
    target_yaw = math.atan2(dy, dx)
    
    if len(robot_pose) >= 7:
        # Isaac Sim conventionally uses [qw, qx, qy, qz]
        qw, qx, qy, qz = robot_pose[3].item(), robot_pose[4].item(), robot_pose[5].item(), robot_pose[6].item()
        # Calculate yaw (Z-axis rotation) from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
    else:
        current_yaw = 0.0
    
    yaw_error = target_yaw - current_yaw
    # Normalize yaw error to [-pi, pi]
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
    
    dist = math.sqrt(dx**2 + dy**2)
    
    # Control logic
    turn_speed = max(min(yaw_error * 1.5, 1.0), -1.0)
    
    if dist < 0.3:
        # Very close to target
        fwd_speed = 0.0
        turn_speed = 0.0
    elif abs(yaw_error) > 0.5:
        # Turn in place if facing the wrong way
        fwd_speed = 0.0
    else:
        # Move forward while adjusting heading
        fwd_speed = max(min(dist, 1.0), 0.1)

    #debugging...    
    #print(f"[DEBUG_NAV] Pos: ({rx:.2f}, {ry:.2f}) | Goal: ({gx:.2f}, {gy:.2f}) | dX: {dx:.2f} dY: {dy:.2f}")
    #print(f"[DEBUG_NAV] Yaw: {current_yaw:.2f} | Tgt Yaw: {target_yaw:.2f} | Err: {yaw_error:.2f} | Fwd: {fwd_speed:.2f} Turn: {turn_speed:.2f}")
        
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
    
    # Override Mapping Configuration for Frontier Exploration
    # We MUST disable floor filtering so the raycaster can clear out empty space.
    if hasattr(env_cfg, "mapping_cfg"):
        env_cfg.mapping_cfg.filter_floor_occupancy = False
        print("[INFO]: Disabled 'filter_floor_occupancy' in environment config for Frontier Exploration.")
        
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        
        obs, _ = env.reset()
        
        step_counter = 0
        target_px = None
        current_frontier_px = None
        blacklist = []
        last_positions = []
        stuck_frames = 0
        recovery_mode = 0
        
        while simulation_app.is_running():
            with torch.inference_mode():
                step_counter += 1
                
                # For simplicity, we process only the first environment (index 0)
                # Ensure we are using the 'policy' dictionary
                if isinstance(obs, dict) and "policy" in obs:
                    obs_dict = obs["policy"]
                elif isinstance(obs, dict):
                    obs_dict = obs
                else:
                    raise ValueError("Expected a dictionary observation containing 'local-map' or 'global-map'")
                
                # Extract map from observations
                map_tensor = obs_dict.get('global-map', None)
                pose_tensor = obs_dict.get('robot-pose', None)
                
                if map_tensor is None or pose_tensor is None:
                    # Fallback to random actions if maps aren't correctly exposed
                    actions = torch.rand((env.unwrapped.num_envs, 5), device=env.unwrapped.device) * 2.0 - 1.0
                    print("what1")
                else:
                    # 1. Convert PyTorch tensor map to NumPy grid for OpenCV/SciPy processing
                    # The environment normalizes the map by dividing by 5.0, so we multiply by 5.0 to restore raw log-odds
                    occ_map_np = map_tensor[0, ..., 0].cpu().numpy() * 5.0
                    
                    # Project 3D map (X, Y, Z) to 2D (X, Y)
                    if occ_map_np.ndim == 3:
                        # Ignore the floor level (Z=0 and Z=1, up to 0.4m) when looking for obstacles, 
                        # because the depth camera raycast will mark the floor as an obstacle since we stopped filtering out floor points!
                        max_val = np.max(occ_map_np[:, :, 2:], axis=2)
                        min_val = np.min(occ_map_np, axis=2)
                        
                        # A 2D column is:
                        # 1. Occupied if any voxel in it is occupied
                        # 2. Free if no voxel is occupied, but AT LEAST ONE voxel is free
                        # 3. Unknown if all voxels are unknown
                        occ_map_2d = np.zeros_like(max_val)
                        occ_map_2d[max_val > 0.1] = 1.0     # Occupied
                        occ_map_2d[(max_val <= 0.1) & (min_val < -0.1)] = -1.0 # Free
                        occ_map_np = occ_map_2d
                        
                    # debugging... because what??
                    if step_counter % 15 == 1:
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(10, 8))
                            plt.imshow(occ_map_np, cmap='coolwarm', origin='lower')
                            plt.colorbar(label='Log-Odds')
                            plt.title(f"2D Occupancy Map (Step {step_counter})")
                            
                            # Mark robot position
                            plt.plot(robot_px[1], robot_px[0], 'g*', markersize=15, label='Robot')
                            
                            # Mark true frontier goal
                            if current_frontier_px is not None:
                                plt.plot(current_frontier_px[1], current_frontier_px[0], 'y*', markersize=15, label='Frontier Goal')
                                
                            # Mark A* lookahead waypoint if exists
                            if target_px is not None:
                                plt.plot(target_px[1], target_px[0], 'rx', markersize=10, label='A* Waypoint')
                                
                            plt.legend()
                            plt.savefig("debug_occ_map.png")
                            plt.close()
                        except Exception as e:
                            print(f"[DEBUG] Failed to save map image: {e}")
                    # ---------------------------
                    
                    # 2. Find frontiers
                    frontiers = find_frontiers(occ_map_np, free_thresh=-0.1, unknown_thresh=0.1)
                    
                    # 3. Cluster frontiers
                    clusters = cluster_frontiers(frontiers)
                    
                    # 4. Select best frontier
                    robot_pose_0 = pose_tensor[0]
                    map_shape = occ_map_np.shape
                    is_global = True #(map_shape[0] == 100) # Global map is 100x153
                    resolution = 0.2
                    
                    if is_global:
                        # Global map is not ego-centric, so compute the robot's pixel coordinates
                        map_origin_x, map_origin_y = -10.5, -12.5
                        rx, ry = robot_pose_0[0].item(), robot_pose_0[1].item()
                        robot_px = (int((rx - map_origin_x) / resolution), int((ry - map_origin_y) / resolution))
                    else:
                        # Local map is ego-centric
                        robot_px = (map_shape[0] // 2, map_shape[1] // 2)
                        
                    # --- STUCK DETECTION ---
                    
                    if len(last_positions) >= 30:
                        dist_moved = math.sqrt((rx - last_positions[-30][0])**2 + (ry - last_positions[-30][1])**2)
                        # If we tried to move but barely made forward progress over 30 frames (0.5s), we are stuck!
                        # (We use 0.3m because physics collisions bounce the robot back, making it seem like it moved)
                        if dist_moved < 0.3 and target_px is not None:
                            stuck_frames += 1
                        else:
                            stuck_frames = 0
                        last_positions.pop(0)
                    last_positions.append((rx, ry))
                    
                    if stuck_frames > 15:
                        print(f"[WARNING] Robot is stuck! Blacklisting current frontier {current_frontier_px} and entering recovery mode...")
                        if current_frontier_px is not None:
                            blacklist.append(current_frontier_px)
                        recovery_mode = 30  # Back up and turn for 30 frames
                        stuck_frames = 0
                        target_px = None    # Force replan after recovery
                        current_frontier_px = None
                        
                    if recovery_mode > 0:
                        # Recovery Action: Drive backwards and rotate to un-wedge from the obstacle
                        action_0 = torch.tensor([[-1.0, 1.0, 0.0, 0.0, 0.0]])
                        actions = action_0.repeat(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
                        env.step(actions)
                        recovery_mode -= 1
                        continue
                    
                    # -----------------------

                    # --- GOAL REACHED DETECTION ---
                    if target_px is not None:
                        if is_global:
                            goal_x = map_origin_x + target_px[0] * resolution
                            goal_y = map_origin_y + target_px[1] * resolution
                            dist_to_target = math.sqrt((goal_x - rx)**2 + (goal_y - ry)**2)
                        else:
                            relative_x = (target_px[0] - robot_px[0]) * resolution
                            relative_y = (target_px[1] - robot_px[1]) * resolution
                            dist_to_target = math.sqrt(relative_x**2 + relative_y**2)
                            
                        if dist_to_target < 0.35:
                            print(f"[INFO] Reached goal/waypoint! Distance: {dist_to_target:.2f}. Replanning...")
                            if target_px == current_frontier_px and current_frontier_px is not None:
                                blacklist.append(current_frontier_px)
                            target_px = None
                    else:
                        print("what2")
                    # ------------------------------
                    
                    # Replan path every 15 steps to prevent Isaac Sim Python thread from stalling/core dumping
                    if step_counter % 15 == 1 or target_px is None:
                        best_frontier_px = select_best_frontier(clusters, robot_px, occupancy_grid=occ_map_np, is_global=is_global, resolution=resolution, blacklist=blacklist)
                        current_frontier_px = best_frontier_px
                        
                        if best_frontier_px is not None:
                            # 5. Plan path with A*
                            obstacle_grid = occ_map_np > 0.1
                            path = a_star(robot_px, best_frontier_px, obstacle_grid)
                            
                            if path and len(path) > 1:
                                # Pick a waypoint a few steps ahead to follow the path smoothly
                                lookahead_idx = min(5, len(path) - 1)
                                target_px = path[lookahead_idx]
                            else:
                                target_px = best_frontier_px
                        else:
                            target_px = None
                            
                    if target_px is not None:
                        if is_global:
                            # Map shape is (X, Y) so target_px is (X_idx, Y_idx)
                            goal_x = map_origin_x + target_px[0] * resolution
                            goal_y = map_origin_y + target_px[1] * resolution
                            action_0 = get_action_towards_goal(robot_pose_0, (goal_x, goal_y))
                        else:
                            # Local map is relative to the robot
                            relative_x = (target_px[0] - robot_px[0]) * resolution
                            relative_y = (target_px[1] - robot_px[1]) * resolution
                            # Fake pose: (0,0) position, and valid quaternion for yaw=0
                            fake_robot_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                            action_0 = get_action_towards_goal(fake_robot_pose, (relative_x, relative_y))
                    else:
                        # No frontiers found (map fully explored or stuck)
                        # Spin to look around
                        action_0 = torch.tensor([[0.0, -1.0, 0.0, 0.0, 0.0]])
                        
                    # Apply action to all environments (or implement per-env logic)
                    actions = action_0.repeat(env.unwrapped.num_envs, 1).to(env.unwrapped.device)

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
