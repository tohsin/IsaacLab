import os
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Old
path_local0 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-17_07-42-02_ppo_gru_128/checkpoints/agent_234000.pt"
#new
path_local1 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-19_21-31-18_ppo_gru_128/checkpoints/agent_369000.pt"
path_local2 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-25_14-04-51_ppo_gru_128/checkpoints/agent_420000.pt"
# path_pretrained = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/Pretrain2026-03-28_01-57-13/checkpoints/agent_201000.pt"
# path_finetune = "scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline/Finetune-SEEIR-Baseline 2026-03-28_22-43-31/checkpoints/agent_156000.pt"
path_pretrained = "scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline/Pretrain-SEEIR-Baseline 2026-04-05_10-46-29/checkpoints/agent_234000.pt"
path_finetune = "scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline/Finetune-SEEIR-Baseline 2026-03-31_09-33-27/checkpoints/agent_312000.pt"

class TrainingConfig_PreTrain:
    optimizer_class = "adam" # "adam" or "muon"
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128
    reset_std = True
    batch_size = 8192 # 8192
    use_attention_fusion = True
    entropy_coef = 3e-5  # Reduced to allow the standard deviation to decrease
    value_loss_scale = 1.0  # Increased to give a stronger signal to the critic
    learning_rate = 2e-5
    init_log_std = 0.0  # Lower initial standard deviation (≈ 0.6)
    use_wandb = True
    global_timesteps = 30_000_000
    scheduler_class =  torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.01,
    }

class TrainingConfig_FineTune(TrainingConfig_PreTrain):
    num_envs = 64
    checkpoint_path = os.path.join(ISAACLAB_ROOT, path_pretrained)
    entropy_coef = 3e-6
    learning_rate = 6e-6
    global_timesteps = 30_000_000
    # init_log_std = -1.0
    use_attention_fusion = True

    
    scheduler_class =  torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.01,
    }
    
class EvaluationConfig:
    optimizer_class = "adam"
    is_eval = True
    headless = False
    # Example path, user should update
    checkpoint_path = os.path.join(ISAACLAB_ROOT, path_pretrained)
    num_envs = 1
    use_wandb = False
    reset_std = False
    use_attention_fusion = True
    batch_size = 8192
    entropy_coef = 3e-7
    learning_rate = 3e-5
    global_timesteps = 30_000_000
    scheduler_class = torch.optim.lr_scheduler.LinearLR
    scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.01,
        "total_iters": -1,
    }
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval")
    save_depth = True

class TrainingConfig_DiverseScratch(TrainingConfig_PreTrain):
    num_envs = 64
    checkpoint_path = None
    entropy_coef = 3e-4  
    learning_rate = 2e-5
    global_timesteps = 40_000_000
    init_log_std = 0.0
    value_loss_scale = 2.0
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.01,
    }

# Select the configuration to use
configs_ = [TrainingConfig_PreTrain(),
 TrainingConfig_FineTune(), 
 EvaluationConfig(), 
 TrainingConfig_DiverseScratch()]   
CONFIG = configs_[0]
