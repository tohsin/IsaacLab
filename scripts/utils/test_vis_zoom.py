import argparse
from isaaclab.app import AppLauncher

# Argument Parser
parser = argparse.ArgumentParser(description="Test Visibility Map with Zoom/Pan.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback
import gymnasium as gym

# --- MONKEY PATCH CONFIG BEFORE IMPORTING ENV ---
import isaaclab_tasks.direct.robot_inspection.run_config as run_config

print("[INFO] Overriding configuration for Visibility Visualization...")
# Force Debug Mode
run_config.cfg_mode = run_config.debug_Cfg
# Enable Visualization
run_config.debug_Cfg.headless = False
run_config.debug_Cfg.visualisation_mode = "visibility"
run_config.debug_Cfg.enable_voxel_visualization = True
run_config.debug_Cfg.num_envs = args_cli.num_envs
# Ensure episode is long enough for our script
run_config.debug_Cfg.min_episode_length = 2000 
run_config.debug_Cfg.max_episode_length = 2000

# Import Environment (Now it will use the modified config)
from isaaclab_tasks.direct.robot_inspection.inspection_env import Isaac3dinspectionEnv
from isaaclab_tasks.direct.robot_inspection.inspection_cfg import Isaac3dinspectionEnvCfg

def main():
    try:
        # Initialize Environment
        env_cfg = Isaac3dinspectionEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = Isaac3dinspectionEnv(cfg=env_cfg, render_mode="rgb_array")

        # Reset
        obs, _ = env.reset()
        print("[INFO] Environment Ready. Starting Scripted Control Sequence...")
        print("[INFO] Watch the Visualization Window (Green = Visible Voxels)")

        # Robot Physics Config for normalizing actions if needed
        # Action space: [left_wheel, right_wheel, pan, tilt, zoom]
        # Range often [-1, 1] mapped to speeds
        
        # Scripted Sequence
        # Each step is roughly 1/120s or defined by dt/decimation. 
        # Typically dt=0.01s * decimation=6 => 0.06s per step.
        # 100 steps ~ 6 seconds.
        
        sequences = [
            ("Wait", 50,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Zoom In", 80, [0.0, 0.0, 0.0, 0.0, 1.0]),  # Zoom/action index 4
            ("Wait", 30,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Pan Right", 60, [0.0, 0.0, 0.5, 0.0, 0.0]), # Pan/action index 2
            ("Wait", 30,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Pan Left", 120,[0.0, 0.0, -0.5, 0.0, 0.0]),
            ("Wait", 30,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Tilt Up", 40,  [0.0, 0.0, 0.0, -0.5, 0.0]), # Tilt/action index 3
            ("Wait", 20,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Tilt Down", 40, [0.0, 0.0, 0.0, 0.5, 0.0]),
            ("Wait", 20,  [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Zoom Out", 80, [0.0, 0.0, 0.0, 0.0, -1.0]),
            ("Wait (End)", 100, [0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        step_count = 0
        
        for name, duration, action_list in sequences:
            print(f"[ACTION] {name} for {duration} steps")
            action_tensor = torch.tensor([action_list] * args_cli.num_envs, device=env.device)
            
            for _ in range(duration):
                env.step(action_tensor)
                step_count += 1
                
        print("[INFO] Sequence Complete.")

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Closing environment...")
        try:
            env.close()
        except:
            pass
        simulation_app.close()

if __name__ == "__main__":
    main()
