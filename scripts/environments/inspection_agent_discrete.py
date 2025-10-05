# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with a discrete action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Discrete action agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Hardcoded discrete actions agent with Isaac Lab environment."""
    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Check if the action space is discrete
    try:
        # Print environment info
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        # Reset environment
        env.reset()

        # Simulate environment
        while simulation_app.is_running():
            # Run everything in inference mode
            with torch.inference_mode():
                
                # Helper functions for discrete actions
                # Note: The action tensor shape for discrete spaces is (num_envs, 1)
                def move_forward_fast():
                    # Action 0: [v_high, ω_zero]
                    return torch.tensor([[0]], device=env.unwrapped.device)
                
                def move_forward_slow():
                    # Action 1: [v_mid, ω_zero]
                    return torch.tensor([[1]], device=env.unwrapped.device)

                def turn_left():
                    # Action 2: [v_mid, ω_high_left]
                    return torch.tensor([[2]], device=env.unwrapped.device)

                def turn_right():
                    # Action 3: [v_mid, ω_high_right]
                    return torch.tensor([[3]], device=env.unwrapped.device)
                
                def rotate_left_in_place():
                    # Action 4: [v_zero, ω_high_left]
                    return torch.tensor([[4]], device=env.unwrapped.device)
                
                def rotate_right_in_place():
                    # Action 5: [v_zero, ω_high_right]
                    return torch.tensor([[5]], device=env.unwrapped.device)

                # A simple, hardcoded navigation sequence using discrete actions
                for i in range(3000):
                    if i < 40:
                        # actions = turn_right()
                        actions = move_forward_fast()
                    elif 40 <= i < 100:
                        actions = turn_right()
                    elif 100 <= i < 320:
                        actions = rotate_right_in_place()
                    elif 320 <= i < 450:
                        actions = move_forward_slow()
                    elif 450 <= i < 580:
                        actions = turn_right()
                    elif 580 <= i < 670:
                        actions = move_forward_fast()
                    elif 580 <= i < 670:
                        actions = turn_right()
                    elif 670 <= i < 800:
                        actions = move_forward_fast()
                    elif 800 <= i < 900:
                        actions = rotate_left_in_place()
                    else:
                        actions = move_forward_slow()

                    # Apply actions and get observations
                    obs, rewards, terminated, truncated, info = env.step(actions)
    
    finally:
        # Close the environment
        env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation app
    simulation_app.close()