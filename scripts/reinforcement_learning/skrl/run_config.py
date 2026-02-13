
from skrl.resources.schedulers.torch import KLAdaptiveRL

class TrainingConfig_1:
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128 # 128
    # checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-02-09_04-38-32_ppo_gru_128/checkpoints/best_agent.pt"
    reset_std = True
    batch_size = 8192 # 8192
    # reset_std = False
    entropy_coef = 3e-7
    learning_rate = 3e-5
    use_wandb = True
    global_timesteps = 10_000_000

    # scheduler_class = torch.optim.lr_scheduler.LinearLR
    # scheduler_kwargs = {
    #     "start_factor": 1.0,     # Start at the full learning_rate
    #     "end_factor": 0.01,      
    #     "total_iters": scheduler_max_steps,
    # } 

    scheduler_class = KLAdaptiveRL
    scheduler_kwargs = {
        "kl_threshold": 0.016,
        "min_lr": 1e-5,
        "max_lr": 3e-5,
        "lr_factor": 1.15
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
    use_wandb = True
    global_timesteps = 50_000_000
    
    scheduler_class = KLAdaptiveRL
    scheduler_kwargs = {
        "kl_threshold": 0.016,
        "min_lr": 1e-5,
        "max_lr": 3e-5,
        "lr_factor": 1.15
    }

class EvaluationConfig:
    is_eval = True
    headless = False
    # Example path, user should update
    checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-01-23_20-06-08_ppo_gru_128/checkpoints/best_agent.pt"
    num_envs = 1
    use_wandb = True

# Select the configuration to use
configs_ = [TrainingConfig_1(), TrainingConfig_2(), EvaluationConfig()]   
CONFIG = configs_[0]
