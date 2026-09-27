# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with PTZ panning and discrete zoom selection."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=60, help="Number of zoom-control steps to run.")

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
    """Agent with rotation, sine-wave PTZ panning, and discrete zoom selection."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = 42
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        
        print("[INFO] Resetting environment...")
        env.reset()

        # run everything in inference mode
        with torch.inference_mode():
            for i in range(args_cli.num_steps):
                if not simulation_app.is_running():
                    break
                        
                # Sweep the inspection camera while rotating the base.
                pan_cmd = math.sin(2 * math.pi * i / 200)
                zoom_cmd = 1.0
                    
                # Actions: [lin_vel, ang_vel, pan, tilt, discrete zoom selector]
                actions_list = [0.0, 0.5, pan_cmd, 0.0, zoom_cmd]
                    
                actions = torch.tensor([actions_list] * args_cli.num_envs, device=env.unwrapped.device)
                env.step(actions)

        expected_level = env.unwrapped.zoom_focal_lengths.numel() - 1
        if not torch.all(env.unwrapped.applied_zoom_levels == expected_level):
            raise RuntimeError(
                "Zoom smoke test ended before every environment reached the telephoto level: "
                f"{env.unwrapped.applied_zoom_levels.tolist()}"
            )
        expected_focal_length = env.unwrapped.zoom_focal_lengths[-1]
        usd_focal_lengths = torch.tensor(
            [camera.GetFocalLengthAttr().Get() for camera in env.unwrapped.camera_prims],
            device=env.unwrapped.device,
        )
        if not torch.allclose(usd_focal_lengths, expected_focal_length.expand_as(usd_focal_lengths)):
            raise RuntimeError(f"Rendered cameras have inconsistent focal lengths: {usd_focal_lengths.tolist()}")

        expected_fx = (
            env.unwrapped.cfg.sensor_cfg.ptz_camera.width
            * expected_focal_length
            / env.unwrapped.cfg.sensor_cfg.ptz_camera.spawn.horizontal_aperture
        )
        rendered_fx = env.unwrapped._ptz_camera.data.intrinsic_matrices[:, 0, 0]
        raycaster_fx = env.unwrapped._raycaster_camera.data.intrinsic_matrices[:, 0, 0]
        if not torch.allclose(rendered_fx, expected_fx.expand_as(rendered_fx)):
            raise RuntimeError(f"Rendered-camera intrinsics are inconsistent: {rendered_fx.tolist()}")
        if not torch.allclose(raycaster_fx, expected_fx.expand_as(raycaster_fx)):
            raise RuntimeError(f"Ray-caster intrinsics are inconsistent: {raycaster_fx.tolist()}")
        print(
            "[INFO] Zoom verification passed: "
            f"all {args_cli.num_envs} cameras reached "
            f"{env.unwrapped.current_focal_lengths[0].item():.1f} mm."
        )
            
    finally:
          env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
