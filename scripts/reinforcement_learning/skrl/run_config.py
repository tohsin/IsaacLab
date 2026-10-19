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


class GroupwiseKLAdaptiveLR(KLAdaptiveRL):
    """KL-adaptive LR that preserves the optimizer groups' relative bounds.

    skrl's scheduler uses one absolute minimum and maximum for every optimizer
    group. This variant expresses those bounds as fractions of each group's
    initial LR, so the lower-LR token-fusion group cannot drift up to the
    policy group's LR.
    """

    def __init__(
        self,
        optimizer,
        kl_threshold=0.016,
        min_lr_scale=0.025,
        max_lr_scale=1.0,
        kl_factor=2.0,
        lr_factor=1.5,
        last_epoch=-1,
        verbose=False,
    ):
        if not 0.0 < min_lr_scale <= max_lr_scale:
            raise ValueError(
                "KL-adaptive LR scales must satisfy "
                f"0 < min_lr_scale <= max_lr_scale; got "
                f"{min_lr_scale} and {max_lr_scale}"
            )

        initial_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs = [lr * min_lr_scale for lr in initial_lrs]
        self.max_lrs = [lr * max_lr_scale for lr in initial_lrs]
        super().__init__(
            optimizer,
            kl_threshold=kl_threshold,
            min_lr=0.0,
            max_lr=float("inf"),
            kl_factor=kl_factor,
            lr_factor=lr_factor,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def step(self, kl=None, epoch=None):
        if kl is None:
            return

        kl = float(kl)
        for group, min_lr, max_lr in zip(
            self.optimizer.param_groups, self.min_lrs, self.max_lrs
        ):
            if kl > self.kl_threshold * self._kl_factor:
                group["lr"] = max(group["lr"] / self._lr_factor, min_lr)
            elif kl < self.kl_threshold / self._kl_factor:
                group["lr"] = min(group["lr"] * self._lr_factor, max_lr)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def get_checkpoint_path(project_name, run_name, checkpoint_type=0, custom_checkpoint=None):
    """
    Helper to automatically fetch the checkpoint path.
    :param project_name: Name of the project/experiment group (e.g. "SEEIR-Baseline")
    :param run_name: Name of the specific run (e.g. "SEEIR-2026-08-12_18-37-46")
    :param checkpoint_type: 0 for 'best_agent.pt', 1 for the latest
        'agent_*.pt', 2 for 'safest_agent.pt', or 3 for a specific checkpoint
    :param custom_checkpoint: Step number (e.g. 129000) or filename
        (e.g. 'agent_129000.pt') when checkpoint_type == 3
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
    elif checkpoint_type == 3:
        if custom_checkpoint is None:
            raise ValueError(
                "custom_checkpoint (e.g. 129000 or 'agent_129000.pt') must be "
                "provided when checkpoint_type=3"
            )
        if isinstance(custom_checkpoint, int) or str(custom_checkpoint).isdigit():
            ckpt_name = f"agent_{int(custom_checkpoint)}.pt"
        else:
            ckpt_name = str(custom_checkpoint)
            if not ckpt_name.endswith(".pt"):
                ckpt_name += ".pt"
        return os.path.join(base_dir, ckpt_name)
    else:
        raise ValueError(
            "checkpoint_type must be 0 (best), 1 (latest), 2 (safest), or 3 (custom)"
        )





Models = {
    # 'Base_model' :{
    #     'path': "Alblation_ATTN_FUS_2026-09-19_10-47-23"
    # },
    # 'Diverse_dataset':
    # {
    #     'path': "Alblation_ATTN_FUS_2026-09-16_14-38-37"
    # },
    # Albaltion study models
    'Paper-Baseline':{
        'path': 'Alblation_ATTN_FUS_2026-09-22_08-11-39',
        'Attention': True,
        'Fusion': 'attention',
    },
    'Paper-Baseline-cntrl':{
            'path': 'Alblation_ATTN_FUS_2026-09-22_08-11-39',
            'Attention': True,
            'Fusion': 'attention',
    },
    # Evaluate the same trained baseline checkpoint with a stricter 95%
    # coverage-success threshold while keeping its results separate.

    # 'MLP_Fusion':{
    #     'path': "Alblation_MLP_FUS_2026-09-24_11-45-07",
    #     'Attention': False,
    #     'Fusion': 'concat_mlp',
    # },
    # Replace the path with the timestamped run name after training, then
    # select Albation('No_MHA_Fusion') for evaluation.
    'No_MHA_Fusion':{
        'path': "Alblation_NO_MHA_FUS_2026-09-25_18-36-35",
        'Attention': False,
        'Fusion': 'token_ffn',
    },
    # You need to go back to the other run config to set the map
    'Map_OCC_Only':{
        'path': "Alblation_MAP_OCC_ONLY_2026-09-28_12-31-28",
        'Attention': True,
        'Fusion': 'attention',
    },
    # Replace the path after the occupancy + visibility model finishes.
    'Map_OCC_Visibility':{
        'path': "Alblation_MAP_OCC_VIS_2026-09-29_12-31-34",
        'Attention': True,
        'Fusion': 'attention',
    },
    'Map_OCC_Visit':{
        'path': "",
        'Attention': True,
        'Fusion': 'attention',
    },
    # Replace the path after the parameter-matched no-GRU model finishes.
    'No_GRU':{
        'path': "Alblation_NO_GRU_2026-10-01_00-26-29",
        'Attention': True,
        'Fusion': 'attention',
        'Temporal': 'feedforward',
    },
    # Replace the path after training. Evaluation also requires
    # eval_Cfg.enable_pt_actuation=False in the environment run config.
    'Fixed_Camera':{
        'path': "Alblation_FIXED_CAMERA_2026-10-01_22-13-59",
        'Attention': True,
        'Fusion': 'attention',
        'Temporal': 'gru',
        'PTActuation': False,
    },
    'Algorithm_Baseline_55M':{
            'path': 'Algorithm_Baseline_70M_2026-10-13_18-39-00',
            'Attention': True,
            'Fusion': 'attention',
    },

     'Algorithm_Baseline_2Depth':{
            'path': 'Algorithm_Baseline_2Depth_2026-10-16_15-30-39',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': False,
    },
     'Algorithm_Baseline_Last':{
            'path': 'Algorithm_Baseline_Last_2026-10-17_10-23-38',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': True,
    },
     'Algorithm_Baseline_Last_129k':{
            'path': 'Algorithm_Baseline_Last_2026-10-17_10-23-38',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': True,
            'Checkpoint': 129000,
    },
     'Algorithm_Baseline_Last_144k':{
            'path': 'Algorithm_Baseline_Last_2026-10-17_10-23-38',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': True,
            'Checkpoint': 144000,
    },
     'Algorithm_Baseline_Last_132k':{
            'path': 'Algorithm_Baseline_Last_2026-10-17_10-23-38',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': True,
            'Checkpoint': 132000,
    },
     'Algorithm_Baseline_Last_153k':{
            'path': 'Algorithm_Baseline_Last_2026-10-17_10-23-38',
            'Attention': True,
            'Fusion': 'attention',
            'ReducedResNet': True,
            'Checkpoint': 153000,
    },
}
class Albation:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = Models[model_name]['path']
        self.fusion_mode = Models[model_name].get(
            'Fusion',
            'attention' if Models[model_name].get('Attention', True) else 'concat_mlp',
        )
        self.temporal_mode = Models[model_name].get('Temporal', 'gru')
        self.pt_actuation = Models[model_name].get('PTActuation', True)
        self.use_reduced_resnet = Models[model_name].get('ReducedResNet', True)
        self.custom_checkpoint = Models[model_name].get('Checkpoint', None)
        # Compatibility attribute for older configuration code.
        self.attention = self.fusion_mode == 'attention'
# CURR_EXPeriment = Albation('Algorithm_Baseline_Last_129k')
#153,144, 132
CURR_EXPeriment = Albation('Algorithm_Baseline_Last_144k')
# CURR_EXPeriment = Albation('Algorithm_Baseline_Last_132k')
# CURR_EXPeriment = Albation('Algorithm_Baseline_Last_153k')
# CURR_EXPeriment = Albation('Algorithm_Baseline_Last')
# CURR_EXPeriment = Albation('Algorithm_Baseline_55M')
# CURR_EXPeriment = Albation('Paper-Baseline')
path_pretrained = get_checkpoint_path(
    project_name="Alblation-Baseline",
    run_name= CURR_EXPeriment.path,
    checkpoint_type=3, # 0: best_agent.pt, 1: latest agent_*.pt, 2: safest_agent.pt, 3: custom Checkpoint
    custom_checkpoint=CURR_EXPeriment.custom_checkpoint,
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
    fusion_mode = "attention"  # "attention", "concat_mlp", or "token_ffn"
    use_attention_fusion = fusion_mode == "attention"
    use_transformer_encoder = True
    # Keep the complete recurrent attention baseline; only PT actuation is
    # ablated in this run.
    temporal_mode = "gru"  # "gru" or "feedforward"
    gru_hidden_size = 512
    gru_num_layers = 1
    temporal_mlp_hidden_size = 1536
    attention_d_model = 512
    # Capacity-matched attention ablation: retain d_model=512 and change only
    # the readout from a learned CLS token to mean pooling over modality tokens.
    attention_pooling = "mean"
    # Set True to revert to the reduced 9-layer ResNet (1 block/stage);
    # False restores the full 15-layer IMPALA ResNet (2 blocks/stage).
    use_reduced_resnet = True
    encoder_res_blocks_per_stage = 1 if use_reduced_resnet else 2
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
    # KL exceeds this guardrail. This is independent of the LR scheduler and
    # catches localized KL spikes that are hidden by the epoch mean.
    kl_threshold = 0.08
    # Canonical order: [linear velocity, angular velocity, pan, tilt]. The
    # fixed-camera action space exposes the leading two entries only.
    init_std = (0.85, 0.85, 0.90, 0.90)
    slice_init_std_to_action_space = True
    manual_std_decay = False
    final_std = 0.3
    std_decay_fraction = 0.90
    use_gsde = True
    use_wandb = True
    global_timesteps = 55_000_000
    # Select "cosine" for time-based decay or "kl_adaptive" to react to policy
    # drift once per PPO learning epoch. The 0.016 KL target matches the
    # semantic-RL-inspection setup.
    lr_scheduler = "cosine"
    kl_adaptive_threshold = 0.010
    kl_adaptive_min_lr_scale = 0.01#0.025
    kl_adaptive_max_lr_scale = 1.0
    scheduler_decay_fraction = 1.0
    if lr_scheduler == "kl_adaptive":
        scheduler_class = GroupwiseKLAdaptiveLR
        scheduler_kwargs = {
            "kl_threshold": kl_adaptive_threshold,
            "min_lr_scale": kl_adaptive_min_lr_scale,
            "max_lr_scale": kl_adaptive_max_lr_scale,
        }
    elif lr_scheduler == "cosine":
        # Run the cosine decay across the full training horizon. Reducing the
        # fraction below 1.0 reaches eta_min earlier and holds it thereafter.
        scheduler_class = CosineAnnealingWithHoldLR
        scheduler_kwargs = {
            "T_max": -1,  # Will be dynamically set
            "eta_min": learning_rate * 0.01,
        }
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
    experiment_name = "Algorithm_Baseline_Last"



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
    temporal_mode = CURR_EXPeriment.temporal_mode
    use_transformer_encoder = True
    attention_d_model = 512
    attention_pooling = "mean"
    # Set True when evaluating checkpoints trained with the reduced 9-layer
    # ResNet (1 block/stage); False uses the full 15-layer ResNet (2 blocks/stage).
    use_reduced_resnet = (
        CURR_EXPeriment.use_reduced_resnet
        if CURR_EXPeriment.use_reduced_resnet is not None
        else False
    )
    encoder_res_blocks_per_stage = 1 if use_reduced_resnet else 2
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
    temporal_mode = "gru"
    attention_d_model = 256
    attention_pooling = "mean"
    use_reduced_resnet = False
    encoder_res_blocks_per_stage = 2
    # The saved Adam state has the historical policy/std two-group layout.
    use_separate_attention_lr = False

# Select the configuration to use
configs_ = [TrainingConfig_PreTrain(), 
            # TrainingConfig_FineTune(),
              EvaluationConfig(),
              LegacyAugust31EvaluationConfig()]
CONFIG = configs_[1]  # Train the full-map, full-MHA, GRU fixed-camera ablation.
#~/evaluate_agent.sh --seed 42 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 43 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 44 --max_episodes 128 --eval-max-episode-steps 1200
# ~/evaluate_agent.sh --seed 45 --max_episodes 128 --eval-max-episode-steps 1200
