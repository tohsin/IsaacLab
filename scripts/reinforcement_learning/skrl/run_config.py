
class TrainingConfig:
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128
    # checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-01-28_04-16-17_ppo_gru_128/checkpoints/best_agent.pt"
    # reset_std = True
    reset_std = False
    use_wandb = True

class EvaluationConfig:
    is_eval = True
    headless = False
    # Example path, user should update
    checkpoint_path = "/home/tosin/Documents/GitHub/IsaacLab/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-01-23_20-06-08_ppo_gru_128/checkpoints/best_agent.pt"
    num_envs = 1
    use_wandb = True

# Select the configuration to use
configs_ = [TrainingConfig(), EvaluationConfig()]   
CONFIG = configs_[0]
