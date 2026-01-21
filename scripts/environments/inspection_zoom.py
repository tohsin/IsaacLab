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
                

                for i in range(3000):
                    if i < 200:
                        actions = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    else:
                        zoom_signal = math.sin(i * 0.05)
                        actions = torch.tensor([[0.0, 0.0, 0.0, 0.0, zoom_signal]], device=env.unwrapped.device)
                  
                  
                    # if i < 100:
                        #     actions = torch.tensor([[0.7, 0.0, 0.0, 0.0]], device=env.unwrapped.device)

                    # else:
                    #     tilt_signal = math.sin(i * 0.05)
                    #     #print(tilt_signal)
                    #     actions = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                    obs, rewards, terminated, truncated, info  = env.step(actions)
                    obs_v = obs['policy']
                #now 
    finally:
          env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
