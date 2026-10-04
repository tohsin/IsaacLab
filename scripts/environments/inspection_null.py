# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# Change only this value to switch the manually driven debug scene. The same
# selection is also available through --environment on the command line.
DEFAULT_ENVIRONMENT = "simple_warehouse"

ISAACLAB_REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_PRESETS = {
    "simple_warehouse": None,
    "usd_explorer_factory": (
        ISAACLAB_REPO_ROOT
        / "assets/environments/nvidia_usd_explorer"
        / "Usd_Explorer/Samples/Examples/2023_2/Factory/Factory.usd"
    ),
    "defect_workshop": (
        ISAACLAB_REPO_ROOT
        / "assets/environments/nvidia_defect_detection/shop.usdc"
    ),
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspection environment null test.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--environment",
    choices=tuple(ENVIRONMENT_PRESETS) + ("custom",),
    default=DEFAULT_ENVIRONMENT,
    help="Environment preset. Use custom together with --environment_usd.",
)
parser.add_argument(
    "--environment_usd",
    type=str,
    default=None,
    help="Local USD/USDA/USDC scene path. This overrides --environment.",
)
parser.add_argument(
    "--environment_scale",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional scale applied to an external environment reference.",
)
parser.add_argument(
    "--environment_offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional XYZ offset applied to an external environment reference.",
)
parser.add_argument(
    "--add_environment_mesh_colliders",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Add static triangle-mesh colliders to external scene meshes that do not already have collision. "
        "Enabled by default for the NVIDIA factory/workshop presets."
    ),
)
parser.add_argument(
    "--keep_procedural_obstacles",
    action="store_true",
    help="Keep the task's generated obstacles when loading an external environment.",
)
parser.add_argument(
    "--inspection_dataset",
    choices=("primitive", "evaluation"),
    default=None,
    help="Override debug_Cfg.inspection_dataset for this process.",
)
parser.add_argument(
    "--inspection_target",
    type=str,
    default=None,
    help="Load one target from the selected inspection dataset.",
)

# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# This utility always needs the interactive configuration. Keeping the choice
# process-local avoids changing the default used by subsequent training runs.
os.environ["ISAACLAB_INSPECTION_RUN_MODE"] = "debug"
if args_cli.inspection_dataset is not None:
    os.environ["ISAACLAB_INSPECTION_DATASET"] = args_cli.inspection_dataset
if args_cli.inspection_target is not None:
    os.environ["ISAACLAB_INSPECTION_TARGET"] = args_cli.inspection_target

# Pass the selection through the environment because the task configuration is
# imported only after Isaac Sim starts. Training and evaluation scripts that do
# not set these variables retain the original Simple Warehouse unchanged.
environment_path_overridden = args_cli.environment_usd is not None
selected_environment_usd = args_cli.environment_usd
if selected_environment_usd is None:
    selected_environment_usd = ENVIRONMENT_PRESETS.get(args_cli.environment)

if args_cli.environment == "custom" and selected_environment_usd is None:
    parser.error("--environment custom requires --environment_usd PATH")

if selected_environment_usd is not None:
    selected_environment_usd = Path(selected_environment_usd).expanduser().resolve()
    if not selected_environment_usd.is_file():
        preset_hint = ""
        if args_cli.environment in {"usd_explorer_factory", "defect_workshop"}:
            preset_hint = (
                " Install it first with: python scripts/environments/"
                f"install_inspection_environment.py {args_cli.environment}"
            )
        parser.error(f"Environment USD not found: {selected_environment_usd}.{preset_hint}")
    os.environ["ISAACLAB_INSPECTION_ENV_USD"] = str(selected_environment_usd)
    selected_environment_name = args_cli.environment
    if environment_path_overridden and selected_environment_name == "simple_warehouse":
        selected_environment_name = "custom"
    os.environ["ISAACLAB_INSPECTION_ENV_NAME"] = selected_environment_name

    add_mesh_colliders = args_cli.add_environment_mesh_colliders
    if add_mesh_colliders is None:
        add_mesh_colliders = environment_path_overridden or args_cli.environment in {
            "usd_explorer_factory",
            "defect_workshop",
            "custom",
        }
    os.environ["ISAACLAB_INSPECTION_ENV_ADD_MESH_COLLIDERS"] = "1" if add_mesh_colliders else "0"

    if args_cli.environment_scale is not None:
        os.environ["ISAACLAB_INSPECTION_ENV_SCALE"] = ",".join(map(str, args_cli.environment_scale))
    if args_cli.environment_offset is not None:
        os.environ["ISAACLAB_INSPECTION_ENV_OFFSET"] = ",".join(map(str, args_cli.environment_offset))

    if not args_cli.keep_procedural_obstacles:
        os.environ["ISAACLAB_INSPECTION_DISABLE_PROCEDURAL_OBSTACLES"] = "1"
else:
    # Make the default debug launch match the normal task environment even if
    # the parent shell happens to contain overrides from an earlier run.
    os.environ["ISAACLAB_INSPECTION_ENV_NAME"] = "simple_warehouse"
    for variable_name in (
        "ISAACLAB_INSPECTION_ENV_USD",
        "ISAACLAB_INSPECTION_ENV_SCALE",
        "ISAACLAB_INSPECTION_ENV_OFFSET",
        "ISAACLAB_INSPECTION_ENV_ADD_MESH_COLLIDERS",
        "ISAACLAB_INSPECTION_DISABLE_PROCEDURAL_OBSTACLES",
    ):
        os.environ.pop(variable_name, None)

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
        action_dim = int(env.unwrapped.cfg.action_space.shape[0])
        keyboard_controller = InspectionKeyboardController(
            device=env.unwrapped.device,
            max_vel_speed=0.8,
            action_dim=action_dim,
        )
        print("[INFO]: Keyboard Controller Initialized.")
        print("[INFO]: Use Arrow Keys (UP/DOWN/LEFT/RIGHT) to move the robot base.")
        print("[INFO]: Use A/S/D/X to pan/tilt the PTZ camera (S: Up, X: Down, A: Left, D: Right).")
        if action_dim == 5:
            print("[INFO]: Hold Q/E to select wide/telephoto zoom; release for the middle zoom level.")

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
                    
                    if "log" in info and "crashes" in info["log"]:
                        crashes = info["log"]["crashes"]
                        if crashes is not None and crashes.numel() > 0 and crashes[0].item() > 0:
                            if "crash_source_counts" in info["log"]:
                                counts = info["log"]["crash_source_counts"][0]
                                source_idx = counts.argmax().item()
                                source_name = env.unwrapped.crash_source_names[source_idx]
                                print(f"\n" + "="*40)
                                print(f"💥 CRASH DETECTED! Source: {source_name} 💥")
                                print("="*40 + "\n")

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
