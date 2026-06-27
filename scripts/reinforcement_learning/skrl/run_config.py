import os
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Old
path_local0 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-17_07-42-02_ppo_gru_128/checkpoints/agent_234000.pt"
#new
path_local1 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-19_21-31-18_ppo_gru_128/checkpoints/agent_369000.pt"
path_local2 = "scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2026-03-25_14-04-51_ppo_gru_128/checkpoints/agent_420000.pt"
path_pretrained = "scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline/SEEIR-2026-06-19_20-58-14/checkpoints/agent_390000.pt"
path_finetune = ""
class TrainingConfig_PreTrain:
    optimizer_class = "adam" # "adam" or "muon"
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128
    reset_std = True
    batch_size = 8192 # 8192
    use_attention_fusion = True
    use_transformer_encoder = False
    use_pose_fourier_encoding = True
    num_pose_frequencies = 4
    activation_fn = "elu"  # "elu" or "silu"
    entropy_coef = 3e-5#0.00006# Moderately increased for better exploration with diverse objects
    value_loss_scale = 1.0 #1.0 
    learning_rate = 6e-5
    std_learning_rate = 3e-4
    grad_clip_norm = 0.7
    init_log_std =  0.0# 0.0 
    manual_std_decay = True
    final_log_std = -3.0#-2.0  # Decays std to ~0.13
    use_gsde = True
    use_wandb = True
    global_timesteps = 50_000_000
    scheduler_class =  torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.1,
    }


class TrainingConfig_FineTune(TrainingConfig_PreTrain):
    num_envs = 128
    checkpoint_path = os.path.join(ISAACLAB_ROOT, path_pretrained)
    entropy_coef = 3e-6
    learning_rate = 6e-5
    global_timesteps = 5_000_000
    reset_std = True
    init_log_std = -1.6  # log(0.2) approx -1.6, giving an std of ~0.2
    
    # Explicitly overriding the scheduler so it evaluates based on the NEW learning rate
    scheduler_kwargs = {
        "T_max": -1,  
        "eta_min": learning_rate * 0.1,  # Decays to 6e-7
    }
    
class EvaluationConfig:
    optimizer_class = "adam"
    is_eval = True
    headless = True
    # Example path, user should update
    checkpoint_path = os.path.join(ISAACLAB_ROOT, path_finetune)
    num_envs = 1
    use_wandb = False
    reset_std = False
    use_attention_fusion = True
    use_pose_fourier_encoding = True
    num_pose_frequencies = 4
    activation_fn = "silu"
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
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval")
    save_depth = True

# Select the configuration to use
configs_ = [TrainingConfig_PreTrain(), TrainingConfig_FineTune(), EvaluationConfig()]   
CONFIG = configs_[0]
