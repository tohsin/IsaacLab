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

def get_pan_tilt_action(step_idx):
    """
    Generates oscillating Pan and Tilt actions.
    
    Args:
        step_idx (int): Current simulation step.
        
    Returns:
        tuple: (pan_action, tilt_action) floats between -1.0 and 1.0
    """
    # Parameters for scanning
    pan_freq = 0.02   # Frequency of Left/Right scan
    pan_amp = 1.0     # Full range (-1 to 1)
    
    tilt_freq = 0.08  # Frequency of Up/Down scan
    tilt_amp = 0.8    # Full range
    
    # Calculate Sine Waves
    pan = pan_amp * math.sin(step_idx * pan_freq)
    tilt = tilt_amp * math.sin(step_idx * tilt_freq)
    
    return pan, 0
ang_vel = -1.0
fwd_vel = 0.8
def turnInPlaceLeft(step_idx, env):
    pan, tilt = get_pan_tilt_action(step_idx)
    return torch.tensor([[0, ang_vel, pan, tilt, 0.0]], device=env.unwrapped.device)
def turnInPlaceRight(step_idx, env):
    pan, tilt = get_pan_tilt_action(step_idx)
    return torch.tensor([[0, -ang_vel, pan, tilt, 0.0]], device=env.unwrapped.device)
def stayInPlace(step_idx, env):
    pan, tilt = get_pan_tilt_action(step_idx)
    return torch.tensor([[0.0, 0.0, pan, tilt, 0.0]], device=env.unwrapped.device)

def moveForward(step_idx, env):
    pan, tilt = get_pan_tilt_action(step_idx)
    return torch.tensor([[fwd_vel, 0.0, pan, tilt, 0.0]], device=env.unwrapped.device)
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
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        
        # reset environment
        env.reset()
        
        print("[INFO] Starting Data Collection with Pan/Tilt Scan...")
        init_steps = 25
        turn_in_place_to_inspect = 40
        T_1 = init_steps + turn_in_place_to_inspect
        turn_in_place_back = 40
        T_2 = T_1 + turn_in_place_back

        while simulation_app.is_running():
            with torch.inference_mode():
                for i in range(1500):
                    if i % 100 == 0:
                        print(f"Step {i}/1500")

                    # Decide Action
                    # turn in place to face object first (for 40 steps)
                    if i < init_steps:
                        actions = turnInPlaceLeft(i, env)
                    # Stall in place to film object 30 steps
                    elif i >= init_steps and i < T_1    :
                        actions = stayInPlace(i, env)
                    # turn right back to continue navigating
                    elif i >= T_1 and i < T_2:
                        actions = turnInPlaceRight(i, env)
                    else:
                        #actions = moveForward(i, env)
                        # Then stay in place and scan? (As per StayInPlace function)
                        # User initialized ang_vel=0.0 in main loop branch if i >= 40 
                        # but calls stayInPlace which uses 0.0, 0.0
                        actions = stayInPlace(i, env)
                        
                    # Step
                    obs, rewards, terminated, truncated, info = env.step(actions)
            
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Closing environment...")
    finally:
        print("[INFO] Finalizing and saving data...")
        env.close()

if __name__ == "__main__":
    main()
