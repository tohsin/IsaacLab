import os
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

class TrainingConfig_1:
    optimizer_class = "adam" # "adam" or "muon"
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128 # 128
    reset_std = True
    batch_size = 8192 # 8192
    use_attention_fusion = True
    entropy_coef = 3e-4
    learning_rate = 2e-5 #3e-5
    init_log_std = 0.0 #0.3 0.0
    use_wandb = True
    global_timesteps = 50_000_000
    scheduler_class =  torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.01,
    }
    


class TrainingConfig_2:
    optimizer_class = "adam"
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 32 # 128
    checkpoint_path = None
    reset_std    = True
    batch_size = 2048 # 8192
    # reset_std = False
    entropy_coef = 3e-7
    learning_rate = 3e-4
    init_log_std = 0.0
    max_log_std = 2.0
    use_wandb = True
    global_timesteps = 50_000_000
    
    scheduler_class = torch.optim.lr_scheduler.LinearLR
    scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.01,
        "total_iters": -1, # Placeholder, will be injected in train script
    }

    scheduler_class = KLAdaptiveRL
    scheduler_kwargs = {
        "kl_threshold": 0.016,
        "min_lr": 1e-6,
        "max_lr": 3e-3,
        "lr_factor": 1.15
    }

class EvaluationConfig:
    optimizer_class = "adam"
    is_eval = True
    headless = False
    # Example path, user should update
    checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-02-27_21-52-56_ppo_gru_128/checkpoints/agent_234000.pt"
    num_envs = 1
    use_wandb = False
    reset_std = False
    batch_size = 8192
    entropy_coef = 3e-7
    learning_rate = 3e-5
    init_log_std = 0.0
    max_log_std = 2.0
    global_timesteps = 30_000_000
    scheduler_class = torch.optim.lr_scheduler.LinearLR
    scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.01,
        "total_iters": -1,
    }
    use_attention_fusion = True
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval")
    save_depth = True

# Select the configuration to use
configs_ = [TrainingConfig_1(), TrainingConfig_2(), EvaluationConfig()]   
CONFIG = configs_[0]
