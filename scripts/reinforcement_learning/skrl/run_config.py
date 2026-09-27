import os
import math
import torch
from skrl.resources.schedulers.torch import KLAdaptiveRL

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

import glob


class CosineAnnealingWithHoldLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine-anneal to ``eta_min``, then hold that floor indefinitely."""

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        return super().get_lr()

    def _get_closed_form_lr(self):
        progress = min(max(self.last_epoch, 0), self.T_max) / self.T_max
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]


def get_checkpoint_path(project_name, run_name, checkpoint_type=0):
    """
    Helper to automatically fetch the checkpoint path.
    :param project_name: Name of the project/experiment group (e.g. "SEEIR-Baseline")
    :param run_name: Name of the specific run (e.g. "SEEIR-2026-08-12_18-37-46")
    :param checkpoint_type: 0 for 'best_agent.pt', 1 for the latest
        'agent_*.pt', or 2 for 'safest_agent.pt'
    """
    base_dir = os.path.join(ISAACLAB_ROOT, "scripts/reinforcement_learning/skrl/logs/skrl", project_name, run_name, "checkpoints")
    
    if checkpoint_type == 0:
        return os.path.join(base_dir, "best_agent.pt")
    elif checkpoint_type == 1:
        pattern = os.path.join(base_dir, "agent_*.pt")
        files = glob.glob(pattern)
        if not files:
            print(f"[WARNING] No agent_*.pt found in {base_dir}")
            return None
        
        # Sort by step number (e.g. agent_1000.pt -> 1000)
        def extract_step(f):
            basename = os.path.basename(f)
            try:
                return int(basename.replace("agent_", "").replace(".pt", ""))
            except ValueError:
                return -1
                
        return max(files, key=extract_step)
    elif checkpoint_type == 2:
        return os.path.join(base_dir, "safest_agent.pt")
    else:
        raise ValueError(
            "checkpoint_type must be 0 (best), 1 (latest), or 2 (safest)"
        )





Models = {
    'Base_model' :{
        'path': "Alblation_ATTN_FUS_2026-09-19_10-47-23"
    },
    'Diverse_dataset':
    {
        'path': "Alblation_ATTN_FUS_2026-09-16_14-38-37"
    },
    # Albaltion study models
    'Paper-Baseline':{
        'path': 'Alblation_ATTN_FUS_2026-09-22_08-11-39',
        'Attention': True,
        'Fusion': 'attention',
    },
    'MLP_Fusion':{
        'path': "Alblation_MLP_FUS_2026-09-24_11-45-07",
        'Attention': False,
        'Fusion': 'concat_mlp',
    },
    # Replace the path with the timestamped run name after training, then
    # select Albation('No_MHA_Fusion') for evaluation.
    'No_MHA_Fusion':{
        'path': "Alblation_NO_MHA_FUS_2026-09-25_18-36-35",
        'Attention': False,
        'Fusion': 'token_ffn',
    }
}
class Albation:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = Models[model_name]['path']
        self.fusion_mode = Models[model_name].get(
            'Fusion',
            'attention' if Models[model_name].get('Attention', True) else 'concat_mlp',
        )
        # Compatibility attribute for older configuration code.
        self.attention = self.fusion_mode == 'attention'
CURR_EXPeriment = Albation('No_MHA_Fusion')
path_pretrained = get_checkpoint_path(
    project_name="Alblation-Baseline",
    run_name= CURR_EXPeriment.path,
    checkpoint_type=0 # 0: best_agent.pt, 1: latest agent_*.pt, 2: safest_agent.pt
)

legacy_aug31_checkpoint = os.path.join(
    ISAACLAB_ROOT,
    "scripts/reinforcement_learning/skrl/logs/skrl/SEEIR-Baseline",
    "SEEIR-2026-08-31_20-03-08/checkpoints/agent_135000.pt",
)

class TrainingConfig_PreTrain:
    optimizer_class = "adam" # "adam" or "muon"
    is_eval = False
    headless = True
    checkpoint_path = None
    num_envs = 128
    reset_std = True
    batch_size = 8192 # 8192
    # Surgical MHA ablation: retain projections, normalization, modality
    # embeddings, Transformer FFNs, mean pooling, and the GRU.
    fusion_mode = "token_ffn"  # "attention", "concat_mlp", or "token_ffn"
    use_attention_fusion = fusion_mode == "attention"
    use_transformer_encoder = True
    attention_d_model = 512
    # Capacity-matched attention ablation: retain d_model=512 and change only
    # the readout from a learned CLS token to mean pooling over modality tokens.
    attention_pooling = "mean"
    encoder_res_blocks_per_stage = 1
    use_pose_fourier_encoding = True
    num_pose_frequencies = 4
    activation_fn = "elu"  # "elu" or "silu"
    entropy_coef =  0.00004
    value_loss_scale = 1.0 #1.0 
    learning_rate = 4e-5
    attention_learning_rate = 2e-5
    use_separate_attention_lr = True
    std_learning_rate = 4e-5
    grad_clip_norm = 0.8
    safety_checkpoint_min_success_rate = 0.90
    safety_checkpoint_interval = 3_000
    # Stop the remaining minibatches in an epoch when the approximate policy
    # KL exceeds this guardrail. This complements the smooth cosine LR decay
    # by catching localized KL spikes that are hidden by the mean KL.
    kl_threshold = 0.08
    # Standard deviations are configured directly in action order:
    # [linear velocity, angular velocity, pan, tilt]. A scalar is also valid
    # and is expanded to every action automatically.
    init_std = (0.85, 0.85, 0.90, 0.90)
    manual_std_decay = False
    final_std = 0.3
    std_decay_fraction = 0.90
    use_gsde = True
    use_wandb = True
    global_timesteps = 45_000_000
    # Run the cosine decay across the full training horizon. Reducing this
    # fraction below 1.0 reaches eta_min earlier and holds it thereafter.
    scheduler_decay_fraction = 1.0
    scheduler_class = CosineAnnealingWithHoldLR
    scheduler_kwargs = {
        "T_max": -1,  # Will be dynamically set
        "eta_min": learning_rate * 0.01,
    }
    experiment_name = "Alblation_NO_MHA_FUS"



class EvaluationConfig:
    optimizer_class = "adam"
    is_eval = True
    deterministic_eval = True
    max_episodes = 512 # Added this so you can set the number of sims here!
    headless = True
    # Example path, user should update
    checkpoint_path = os.path.join(ISAACLAB_ROOT, path_pretrained)
    num_envs = 1
    use_wandb = False
    reset_std = False
    fusion_mode = CURR_EXPeriment.fusion_mode
    use_attention_fusion = fusion_mode == "attention"
    use_transformer_encoder = True
    attention_d_model = 512
    attention_pooling = "mean"
    encoder_res_blocks_per_stage = 1
    use_separate_attention_lr = True
    use_pose_fourier_encoding = True
    num_pose_frequencies = 4
    activation_fn = "elu"
    batch_size =  8192
    entropy_coef = 3e-7
    learning_rate = 3e-5
    kl_threshold = 0.0
    init_std = 1.0
    max_log_std = 2.0
    global_timesteps = 30_000_000
    scheduler_class = torch.optim.lr_scheduler.LinearLR
    scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.01,
        "total_iters": -1,
    }
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval")
    save_depth = False
    experiment_name = ""


class LegacyAugust31EvaluationConfig(EvaluationConfig):
    """Architecture-compatible evaluation for SEEIR-2026-08-31_20-03-08."""

    checkpoint_path = legacy_aug31_checkpoint
    fusion_mode = "attention"
    use_attention_fusion = True
    attention_d_model = 256
    attention_pooling = "mean"
    encoder_res_blocks_per_stage = 2
    # The saved Adam state has the historical policy/std two-group layout.
    use_separate_attention_lr = False

# Select the configuration to use
configs_ = [TrainingConfig_PreTrain(), 
            # TrainingConfig_FineTune(),
              EvaluationConfig(),
              LegacyAugust31EvaluationConfig()]
CONFIG = configs_[1]  # Train the no-MHA token-FFN ablation
#~/evaluate_agent.sh --seed 42 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 43 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 44 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 45 --max_episodes 128 --eval-max-episode-steps 1200
