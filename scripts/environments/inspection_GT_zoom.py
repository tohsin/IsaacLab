# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
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

"""Rest everything follows."""
import math
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.robot_inspection import run_config

# Force Recording Mode
# run_config.cfg_mode = run_config.modes[4] # record_point_cloud_Cfg
# run_config.cfg_mode.data_recording_path = "data/recorded_point_clouds"
# run_config.cfg_mode.save_depth = True
# step = 0.0465sec/step


def main():
    """Random actions agent with Isaac Lab environment."""
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

        # print info (this is vectorized environment)
        #cartpole
        #INFO]: Gym observation space: Box(-inf, inf, (1, 100, 100, 3), float32)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment

        env.reset()
        # 
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                

                import math
                
                # 0.7 * 0.07
                # Orb
                Tilt_T = 70
                move_forward_start = 22
                T_1 = Tilt_T + move_forward_start
                turn_in_place = 110
                T_2 = T_1 + turn_in_place
                move_fwd = 75
                T_3 = T_2 + move_fwd
                T_4 = T_3 + turn_in_place
                T_5 = T_4 + move_fwd
                T_6 = T_5 + turn_in_place
                T_7 = T_6 + move_fwd
                T_8 = T_7 + turn_in_place
                print(f"[INFO]: Tilt_T: {Tilt_T}")
                print(f"[INFO]: T_1: {T_1}")
                print(f"[INFO]: T_2: {T_2}")
                print(f"[INFO]: T_3: {T_3}")
                print(f"[INFO]: T_4: {T_4}")
                print(f"[INFO]: T_5: {T_5}")
                print(f"[INFO]: T_6: {T_6}")
                print(f"[INFO]: T_7: {T_7}")
                print(f"[INFO]: T_8: {T_8}")
                
                fwd_speed = 0.7
                turn_speed = -0.8
                for i in range(800):
                    # original sine wave
                    zoom_val = math.sin(i * 0.05)
                    # zoom_val = math.sin(i * 0.02) * 0.5
                    # spend some time tilting to the side
                    if i < Tilt_T:
                        actions = torch.tensor([[0.0, 0.0, -1.0, 0.03, zoom_val]], device=env.unwrapped.device)
                    # forward
                    elif i >= Tilt_T and i < T_1:
                        actions = torch.tensor([[fwd_speed, 0.0, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    # Turn in Place
                    elif i >= T_1 and i < T_2: 
                        actions = torch.tensor([[0.0, turn_speed, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    # Forward
                    elif i >= T_2 and i < T_3:
                        actions = torch.tensor([[fwd_speed, 0.0, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    # Turn in Place
                    elif i >= T_3 and i < T_4:
                        actions = torch.tensor([[0.0, turn_speed, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    # Forward
                    elif i >= T_4 and i < T_5:
                        actions = torch.tensor([[fwd_speed, 0.0, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    elif i >= T_5 and i < T_6:
                        # Turn
                        actions = torch.tensor([[0.0, turn_speed, 0.0,  0, zoom_val]], device=env.unwrapped.device)
                    elif i >= T_6 and i < T_7:
                        # Forward
                        actions = torch.tensor([[fwd_speed, 0.0, 0.0, 0, zoom_val]], device=env.unwrapped.device)
                    elif i >= T_7 and i < T_8:
                        # Turn
                        actions = torch.tensor([[0.0, turn_speed, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)
                    else:
                        actions = torch.tensor([[fwd_speed, 0.0, 0.0, 0.0, zoom_val]], device=env.unwrapped.device)

                    obs, rewards, terminated, truncated, info  = env.step(actions)
                    
                    if terminated.any() or truncated.any():
                        print("[INFO] Episode terminated/truncated. Checking for Point Cloud in info...")
                        if "log" in info and "point_cloud" in info["log"]:
                            pc = info["log"]["point_cloud"]
                            # Retrieve single env point cloud (might be in a list or tensor depending on impl)
                            # In _reset_idx it's assigned directly: self.extras["log"]["point_cloud"] = full_cloud
                            if isinstance(pc, torch.Tensor):
                                pc = pc.cpu().numpy()
                            
                            print(f"[INFO] Found Point Cloud in info with {pc.shape[0]} points. Saving...")
                            filename = "inspection_GT_cloud_reset.ply"
                            import numpy as np
                            with open(filename, 'w') as f:
                                f.write("ply\n")
                                f.write("format ascii 1.0\n")
                                f.write(f"element vertex {len(pc)}\n")
                                f.write("property float x\n")
                                f.write("property float y\n")
                                f.write("property float z\n")
                                f.write("end_header\n")
                                np.savetxt(f, pc, fmt="%.6f")
                            print(f"[SUCCESS] Saved to {filename}")
                            # Optional: Break if we just wanted one episode
                            break
                        else:
                            print("[WARNING] Terminated but 'point_cloud' not found in info['log'].")

                    obs_v = obs['policy']
                #now 
            print("[INFO] Episode actions completed. Exiting simulation to process recorded data.")
            break
            
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Closing environment...")
    finally:
        print("[INFO] Finalizing...")
        env.close()
        print("[INFO] Data collection complete. Run the generation script to create the point cloud.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
