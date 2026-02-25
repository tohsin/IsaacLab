
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL

class TrainingConfig_1:
    is_eval = False
    headless = True
    checkpoint_path = None
    # checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-02-15_15-11-05_ppo_gru_128/checkpoints/agent_155000.pt"
    num_envs = 128 # 128
    # checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-02-09_04-38-32_ppo_gru_128/checkpoints/best_agent.pt"
    reset_std = True
    batch_size = 8192 # 8192

    entropy_coef = 1e-7
    learning_rate = 3e-5
    init_log_std = 0.3 #0.0
    max_log_std = 0.0 # 2.0
    use_wandb = True
    global_timesteps = 30_000_000

    scheduler_class = torch.optim.lr_scheduler.LinearLR
    scheduler_kwargs = {
        "start_factor": 1.0,     # Start at the full learning_rate
        "end_factor": 0.01,      
        "total_iters": -1,
    } 


class TrainingConfig_2:
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 32 # 128
    checkpoint_path = None
    reset_std    = True
    batch_size = 2048 # 8192
    # reset_std = False
    entropy_coef = 3e-7
    learning_rate = 3e-5
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

    # scheduler_class = KLAdaptiveRL
    # scheduler_kwargs = {
    #     "kl_threshold": 0.016,
    #     "min_lr": 1e-5,
    #     "max_lr": 3e-5,
    #     "lr_factor": 1.15
    # }

class EvaluationConfig:
    is_eval = True
    headless = True
    # Example path, user should update
    checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-02-22_21-06-49_ppo_gru_128/checkpoints/agent_130000.pt"
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

# Select the configuration to use
configs_ = [TrainingConfig_1(), TrainingConfig_2(), EvaluationConfig()]   
CONFIG = configs_[0]
