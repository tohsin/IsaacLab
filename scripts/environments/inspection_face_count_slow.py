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
run_config.cfg_mode = run_config.modes[3] 
run_config.cfg_mode.data_recording_path = "data/test_3dgs_collection_v2"
run_config.cfg_mode.use_wandb = False


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
        # 
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                

                for i in range(800):
                    if i ==570:
                        print("done with filming")
                    if i < 30:
                        #forward
                        actions = torch.tensor([[1.0, 0.0, -1.0, 0.0]], device=env.unwrapped.device)
                    elif i>=30 and i < 110: 
                        #print Turn
                        actions = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=110 and i <180:
                        #print Forward
                        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=180 and i < 250:
                        # print turn
                        actions = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=250 and i <320:
                        # forward
                        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=320 and i <400:
                        # turn
                        actions = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=400 and i <480:
                        # forward
                        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    elif i>=480 and i <560:
                        # turn
                        actions = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=env.unwrapped.device)
                    
                    else:
                        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    obs, rewards, terminated, truncated, info  = env.step(actions)
                    obs_v = obs['policy']
                #now 
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Closing environment...")
    finally:
        print("[INFO] Finalizing and saving data...")
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
