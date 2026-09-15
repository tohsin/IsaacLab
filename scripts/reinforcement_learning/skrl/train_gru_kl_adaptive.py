import argparse
import json
import os
import sys

if "LOCAL_RANK" in os.environ:
    # Save the real local rank for our monkey patch later
    os.environ["REAL_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
    
    # 1. Restrict this process to only see its assigned physical GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    
    # 2. Trick Omniverse and SKRL into thinking they are running on device 0
    os.environ["LOCAL_RANK"] = "0"

# Keep every worker reproducible while preventing the two GPU processes from
# generating identical reset trajectories. RANK is globally unique under
# torchrun; REAL_LOCAL_RANK is the single-node fallback preserved above.
PROCESS_RANK = int(os.environ.get("RANK", os.environ.get("REAL_LOCAL_RANK", "0")))

# Remove --loca_rank to prevent AppLauncher from reading it
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank") and not arg.startswith("--local-rank")]

import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import warnings
# from heavyball import ForeachMuon
# import heavyball.utils
# torch.autograd.set_detect_anomaly(True)

# conda install -c conda-forge gcc=12 -y
from isaaclab.app import AppLauncher
from run_config import CONFIG
is_eval = CONFIG.is_eval

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
#multi GPU Code
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--num_envs", type=int, default=CONFIG.num_envs, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path, help="Path to checkpoint to resume training from.")
parser.add_argument("--reset_std", action="store_true", default=CONFIG.reset_std, help="Reset the standard deviation to initial value (promotes exploration).")
parser.add_argument("--max_episodes", type=int, default=getattr(CONFIG, "max_episodes", 20), help="Maximum number of episodes to run in evaluation mode.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Override the configured training/evaluation seed.",
)
parser.add_argument(
    "--eval_max_episode_steps",
    "--eval-max-episode-steps",
    "--max_episode_steps",
    dest="eval_max_episode_steps",
    type=int,
    default=None,
    help="Override the evaluation episode horizon without modifying run_config.py.",
)
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
_use_wandb = CONFIG.use_wandb
_headless = CONFIG.headless
args_cli.enable_cameras =  True
args_cli.headless = _headless
#multi GPU Code


# monkey-patch SimulationApp to fix Vulkan interop mismatch
from isaacsim import SimulationApp
_original_init = SimulationApp.__init__

def _patched_init(self, launch_config=None, *args, **kwargs):
    if launch_config is not None and "REAL_LOCAL_RANK" in os.environ:
        real_rank = int(os.environ["REAL_LOCAL_RANK"])
        # Force Vulkan to use the actual physical GPU
        launch_config["active_gpu"] = real_rank
        # Force Physics/CUDA to use device 0 (which maps to the physical GPU via CUDA_VISIBLE_DEVICES)
        launch_config["physics_gpu"] = 0
    _original_init(self, launch_config, *args, **kwargs)

SimulationApp.__init__ = _patched_init

# launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ppo_rnn_custom import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict
import gymnasium as gym
import isaaclab_tasks
from encoder import ResnetEncoder, Resnet3DEncoder
from isaaclab_tasks.utils import parse_env_cfg
from muon import Muon

# sys.argv.append("--headless")
sys.argv.append("--enable_cameras")
BASE_SEED = int(
    args_cli.seed if args_cli.seed is not None else getattr(CONFIG, "seed", 42)
)
WORKER_SEED = BASE_SEED + PROCESS_RANK
set_seed(WORKER_SEED, deterministic=True)
print(f"[INFO] Distributed rank {PROCESS_RANK} using seed {WORKER_SEED}")
# for some reason changing the clip actionsvarialbe to true in thr training script causes this error

class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # Transformer-style log-linear spacing, but scaled UP for continuous values in [-1, 1]
        # (Standard Transformer PE scales down because positions are large integers like 0, 1, ..., 1000)
        import math
        frequencies = torch.exp(torch.arange(num_frequencies) * (math.log(10000.0) / max(1, num_frequencies - 1)))
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        scaled_x = x.unsqueeze(-1) * self.frequencies
        sin_x = torch.sin(scaled_x)
        cos_x = torch.cos(scaled_x)
        encoded = torch.cat([x.unsqueeze(-1), sin_x, cos_x], dim=-1)
        return encoded.view(*x.shape[:-1], self.input_dim * (1 + self.num_frequencies * 2))

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                observation_space,
                action_space,
                device,
                cfg,
                clip_actions=False,
                # clip_log_std=True, min_log_std=-20, max_log_std=2,
                init_log_std = getattr(CONFIG, "init_log_std", 0.0),
                clip_log_std=True, min_log_std=-20, max_log_std=getattr(CONFIG, "max_log_std", 2.0),
                num_envs=1,
                sequence_length=32,
                _hidden_size=128,
                _hidden_size_gru=256,
                use_attention_fusion=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        DeterministicMixin.__init__(self, False)
        self.init_log_std = init_log_std
        self.cfg = cfg
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self._hidden_size = _hidden_size
        self._hidden_size_gru = _hidden_size_gru
        self.use_attention_fusion = use_attention_fusion

        camera_space = observation_space.spaces["cameras"]
        self.camera_shape = camera_space.shape
        self.camera_flat_size = np.prod(self.camera_shape).item()

        map_space = observation_space.spaces["local-map"]
        self.map_shape = map_space.shape
        self.map_flat_size = np.prod(self.map_shape).item()

        robot_pose_space = observation_space.spaces["robot-pose"]
        self.robot_pose_dim = robot_pose_space.shape[0]

        camera_shape_permuted = (self.camera_shape[-1], *self.camera_shape[:-1])
        camera_space_permuted = gym.spaces.Box(low=0, high=255, shape=camera_shape_permuted)

        map_shape_permuted = (self.map_shape[-1], *self.map_shape[:-1])
        map_space_permuted = gym.spaces.Box(low=-np.inf, high=np.inf, shape=map_shape_permuted)

        self.camera_encoder = ResnetEncoder(self.cfg, camera_space_permuted)
        self.map_encoder = Resnet3DEncoder(self.cfg, map_space_permuted, type="occupancy")
       
        print(f"DEBUG: camera_shape: {self.camera_shape}")
        print(f"DEBUG: map_shape: {self.map_shape}")
        print(f"DEBUG: robot_pose_dim: {self.robot_pose_dim}")

        #self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim + map_cnn_output_dim
        camera_features_size = self.camera_encoder.get_out_size()
        map_features_size = self.map_encoder.get_out_size()

        self.use_pose_fourier_encoding = getattr(CONFIG, "use_pose_fourier_encoding", False)
        self.num_pose_frequencies = getattr(CONFIG, "num_pose_frequencies", 4)
        
        if self.use_pose_fourier_encoding:
            self.pose_encoder = ContinuousPositionalEncoding(self.robot_pose_dim, self.num_pose_frequencies)
            self.encoded_pose_dim = self.robot_pose_dim * (1 + self.num_pose_frequencies * 2)
            print(f"[INFO] Using Pose Fourier Encoding (Freqs: {self.num_pose_frequencies}, Dim: {self.robot_pose_dim} -> {self.encoded_pose_dim})")
        else:
            self.pose_encoder = nn.Identity()
            self.encoded_pose_dim = self.robot_pose_dim

        self.combined_features_size = camera_features_size + self.encoded_pose_dim + map_features_size
        
        if self.use_attention_fusion:
            # --- ATTENTION-BASED FUSION ---
            print("[INFO] Using Attention-Based Fusion")
            self.d_model = int(getattr(CONFIG, "attention_d_model", 512))
            self.attention_pooling = str(
                getattr(CONFIG, "attention_pooling", "cls")
            ).lower()
            if self.attention_pooling not in {"cls", "mean"}:
                raise ValueError(
                    "attention_pooling must be either 'cls' or 'mean', got "
                    f"{self.attention_pooling!r}"
                )
            
            # Projectors
            self.camera_proj = nn.Linear(camera_features_size, self.d_model)
            self.map_proj = nn.Linear(map_features_size, self.d_model)
            self.pose_proj = nn.Linear(self.encoded_pose_dim, self.d_model)
            
            # Token Normalization (avoids clamping and stabilizes attention)
            self.token_norm = nn.LayerNorm(self.d_model)
            
            # Modality embeddings
            # self.modality_embeddings = nn.Parameter(torch.randn(1, 3, self.d_model))
            self.modality_embeddings = nn.Parameter(torch.randn(1, 3, self.d_model) * 0.02)

            if self.attention_pooling == "cls":
                # Learned fusion token. Its attended representation is passed
                # to the GRU instead of averaging modalities equally.
                self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            
            self.use_transformer_encoder = getattr(CONFIG, "use_transformer_encoder", False)
            
            if self.use_transformer_encoder:
                # Transformer Encoder
                # We add norm_first=True (Pre-LN) which is much more stable for RL
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model, 
                    nhead=4, 
                    dim_feedforward=512, 
                    batch_first=True,
                    activation='gelu',
                    dropout=0.0,
                    norm_first=True
                )
                self.sensor_attention = nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=2,
                    norm=nn.LayerNorm(self.d_model) # Final layer norm
                )
            else:
                # Simple Multi-Head Attention for stable feature fusion
                self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True, dropout=0.0)
                self.mha_norm = nn.LayerNorm(self.d_model)
            
            print(
                f"[INFO] Attention configuration: d_model={self.d_model}, "
                f"pooling={self.attention_pooling}"
            )
            self.gru_input_size = self.d_model
        else:
            act_str = getattr(CONFIG, "activation_fn", "elu").lower()
            def get_activation():
                return nn.SiLU() if act_str == "silu" else nn.ELU()

            # --- MLP-BASED FUSION (Original) ---
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.combined_features_size, 2048),
                get_activation(),
                nn.Linear(2048, 1024),
                get_activation(),
                nn.Linear(1024, 512),
                get_activation()
            )
            
            self.gru_input_size = 512 # Output of the feature MLP
        self.gru_hidden_size = 512 #H output size of GRU
        self.gru_num_layers = 1
        # print(f"DEBUG: gru_input_size: {self.gru_input_size}")

        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)
        #output heads

        act_str = getattr(CONFIG, "activation_fn", "elu").lower()
        def get_activation():
            return nn.SiLU() if act_str == "silu" else nn.ELU()

        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 1024),
            get_activation(),
            nn.Linear(1024, 512),
            get_activation(),
            nn.Linear(512, 256),
            get_activation(),
            nn.Linear(256, self.num_actions ),
            nn.Tanh()
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 1024),
            get_activation(),
            nn.Linear(1024, 512),
            get_activation(),
            nn.Linear(512, 256),
            get_activation(),
            nn.Linear(256, 1)
        )
        # Action Head, MU and STD
        self.log_std_parameter = nn.Parameter(self.init_log_std * torch.ones(self.num_actions))
        if getattr(CONFIG, "manual_std_decay", False):
            self.log_std_parameter.requires_grad = False


    def get_specification(self) -> dict:
        return {
                "rnn": {
                        "sequence_length": self.sequence_length,
                        "sizes": [(self.gru_num_layers, self.num_envs, self.gru_hidden_size)],
                    }
                }
            
    def unflatten_observations(self, flat_obs):
        """
        Manually unflatten the observation tensor back to camera and robot pose components.
        """
        batch_size = flat_obs.shape[0]
        
        # print(f"DEBUG: Unflattening obs with shape: {flat_obs.shape}")
        # print(f"DEBUG: Expected total size: {self.total_obs_size}")
        
        # Verify the flattened observation has the expected size
        # Split camera and robot pose data
        cam_end = self.camera_flat_size
        map_end = cam_end + self.map_flat_size

        pose_start = cam_end

        # pose_start, pose_end = cam_end, cam_end + self.robot_pose_dim
        # map_start = pose_end

        camera_flat = flat_obs[:, :cam_end]
        map_flat = flat_obs[:, cam_end:map_end]
        robot_pose = flat_obs[:, map_end:]

        # print(f"DEBUG: camera_flat shape: {camera_flat.shape}")
        # print(f"DEBUG: robot_pose shape: {robot_pose.shape}")
        
        # Debug: Print some values to verify splitting (especially useful with all-ones robot pose)
        # print(f"DEBUG: First few camera values: {camera_flat[0, :5]}")
        # print(f"DEBUG: Robot pose values: {robot_pose[0]}")
        # print(f"DEBUG: First few map values: {map_flat[0, :5]}")
        
        # Reshape camera data from flat to (batch, height, width, channels)
        camera_obs = camera_flat.view(-1, *self.camera_shape)
        map_obs = map_flat.view(-1, *self.map_shape)

        # print(f"DEBUG: camera_obs final shape: {camera_obs.shape}")

        return camera_obs, map_obs,  robot_pose

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        camera_obs, local_map, robot_pose,  = self.unflatten_observations(states)

        camera_obs_permuted = camera_obs.permute(0, 3, 1, 2)
        local_map_permuted = local_map.permute(0, 4, 1, 2, 3)

        # ---- DEBUG CHECKS ----
        if torch.isnan(states).any():
            print(f"[MODEL DEBUG] NaN detected in 'states' input! Size: {states.shape}")
        if states.abs().max() > 100:
            print(f"[MODEL DEBUG] 'states' contains extremely large values! Max: {states.abs().max().item()}")
        if torch.isnan(camera_obs_permuted).any() or camera_obs_permuted.abs().max() > 100:
            print(f"[MODEL DEBUG] Issue in camera_obs! NaN: {torch.isnan(camera_obs_permuted).any().item()}, Max: {camera_obs_permuted.abs().max().item()}")
        if torch.isnan(local_map_permuted).any() or local_map_permuted.abs().max() > 100:
            print(f"[MODEL DEBUG] Issue in local_map! NaN: {torch.isnan(local_map_permuted).any().item()}, Max: {local_map_permuted.abs().max().item()}")
        # ----------------------

        camera_features = self.camera_encoder(camera_obs_permuted)
        map_features = self.map_encoder(local_map_permuted)
        encoded_pose = self.pose_encoder(robot_pose)
        
        if self.use_attention_fusion:
            # Sanity checks before projection
            if not torch.isfinite(camera_features).all(): print("[MODEL DEBUG] NaN/Inf in camera_features!")
            if not torch.isfinite(map_features).all(): print("[MODEL DEBUG] NaN/Inf in map_features!")
            if not torch.isfinite(encoded_pose).all(): print("[MODEL DEBUG] NaN/Inf in encoded_pose!")

            # Project to d_model and shape into tokens: [batch_size, 1, d_model]
            cam_tok = self.camera_proj(camera_features).unsqueeze(1)
            map_tok = self.map_proj(map_features).unsqueeze(1)
            pose_tok = self.pose_proj(encoded_pose).unsqueeze(1)
            
            # Sequence of tokens: [batch_size, 3, d_model]
            tokens = torch.cat([cam_tok, map_tok, pose_tok], dim=1)
            
            # assert torch.isfinite(tokens).all(), "NaN/Inf right after proj+cat"

            # Apply LayerNorm to stabilize values before adding embeddings
            tokens = self.token_norm(tokens)

            # Add modality embeddings so it knows which token is which
            tokens = tokens + self.modality_embeddings

            if self.attention_pooling == "cls":
                # Prepend one learned query that can combine the modalities
                # differently for every observation.
                cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
                tokens = torch.cat((cls_token, tokens), dim=1)
            
            if self.use_transformer_encoder:
                # Safety net: clamp extreme outliers gracefully without affecting nominal gradients
                # tokens = torch.clamp(tokens, min=-20.0, max=20.0)
                
                # Cross-Sensor Attention
                import torch.nn.attention as attn
                if hasattr(attn, 'sdpa_kernel'):
                    with attn.sdpa_kernel(attn.SDPBackend.MATH):
                        attended_tokens = self.sensor_attention(tokens)
                else:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        attended_tokens = self.sensor_attention(tokens)
            else:
                # Use a simple Multi-Head Self-Attention layer instead of a deep Transformer
                # This is more stable for RL and prevents feature explosion without needing clamps
                import torch.nn.attention as attn
                if hasattr(attn, 'sdpa_kernel'):
                    with attn.sdpa_kernel(attn.SDPBackend.MATH):
                        attn_output, _ = self.mha(tokens, tokens, tokens, need_weights=False)
                else:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        attn_output, _ = self.mha(tokens, tokens, tokens, need_weights=False)
                
                # Residual connection + LayerNorm
                attended_tokens = self.mha_norm(tokens + attn_output)
            
            if self.attention_pooling == "cls":
                fusion_features = attended_tokens[:, 0]
            else:
                # Legacy August 31 checkpoint behavior: average the attended
                # camera, occupancy-map, and pose tokens.
                fusion_features = attended_tokens.mean(dim=1)
        else:
            combined_features = torch.cat((camera_features, map_features, encoded_pose), dim=1)
            fusion_features = self.feature_mlp(combined_features)

        if self.training:
            # just return dummy action to debug sim
            # return torch.zeros((self.num_envs, self.num_actions), device=self.device), {"rnn": [hidden_states]}
            rnn_input = fusion_features.view(-1, self.sequence_length, fusion_features.shape[-1])
            hidden_states = hidden_states.view(self.gru_num_layers, -1, self.sequence_length, self.gru_hidden_size)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)

                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i+1]
                    rnn_output, hidden_states = self.gru(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    # Clone hidden states before modifying them to avoid breaking autograd BPTT
                    hidden_states = hidden_states.clone()
                    hidden_states[:, terminated[:, i1 - 1], :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            rnn_input = fusion_features.unsqueeze(1)
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)


        #flatten  rnn output
        # flat_gru_output = gru_output.reshape(-1, self.gru_hidden_size)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        if role == "policy":
            mean_actions = self.policy_head(rnn_output)
            log_std = self.log_std_parameter.expand_as(mean_actions)
            return mean_actions, log_std, {"rnn": [hidden_states]}
        elif role == "value":
            value_estimate = self.value_head(rnn_output)
            return value_estimate, {"rnn": [hidden_states]}


#multi GPU code
if args_cli.distributed:
    # Since we use CUDA_VISIBLE_DEVICES, each process only sees one GPU, which is cuda:0
    args_cli.device = "cuda:0"

env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)

env_cfg.seed = WORKER_SEED
env = gym.make(args_cli.task, cfg=env_cfg)
if args_cli.eval_max_episode_steps is not None:
    if not is_eval:
        raise ValueError("--eval-max-episode-steps is only valid in evaluation mode")
    if args_cli.eval_max_episode_steps <= 0:
        raise ValueError("--eval-max-episode-steps must be greater than zero")
    horizon_env = env.unwrapped if hasattr(env, "unwrapped") else env
    horizon_env.curriculum.min_episode_length_limit = args_cli.eval_max_episode_steps
    horizon_env.curriculum.max_episode_length_limit = args_cli.eval_max_episode_steps
    print(
        "[INFO] Evaluation episode horizon overridden to "
        f"{args_cli.eval_max_episode_steps} steps"
    )
env = wrap_env(env)

device = env.device
# assume num env is 16
TOTAL_BATCH_SIZE = CONFIG.batch_size #8192# 2048
sequence_length = 32
# rollout_length = TOTAL_BATCH_SIZE // env.num_envs
if torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
else:
    world_size = 1
rollout_length = TOTAL_BATCH_SIZE // (env.num_envs * world_size)

from skrl.memories.torch import RandomMemory
memory = RandomMemory(memory_size=rollout_length, num_envs=env.num_envs, device=device)
model_config = {
        "nonlinearity": getattr(CONFIG, "activation_fn", "elu").lower(),
        "encoder_conv_architecture": "resnet_impala",
        "encoder_conv_mlp_layers": [256],
        "encoder_conv_map_occupancy_architecture": "resnet",
        "encoder_conv_map_occupancy_mlp_layers": [128, 128],
        "encoder_res_blocks_per_stage": int(
            getattr(CONFIG, "encoder_res_blocks_per_stage", 1)
        ),
    }
models = {}
models['policy'] = Shared(env.observation_space,
                            env.action_space,
                            env.device,
                            cfg=model_config,
                            num_envs=env.num_envs,
                            sequence_length=sequence_length,
                            use_attention_fusion=getattr(CONFIG, "use_attention_fusion", False))
models['value'] = models["policy"]  # Shared(env.observation_space, env.action_space, env.device)
total_timesteps = CONFIG.global_timesteps // (env.num_envs * world_size)

cfg = PPO_DEFAULT_CONFIG.copy()
# warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
# heavyball.utils.compile_mode = None
cfg["rollouts"] = rollout_length  # memory_size
cfg["learning_epochs"] = 3  # Reduced from 4 to limit policy drift per rollout
cfg["mini_batches"] = 8   # 16 horizon_length * num_actors / minibatch_size   8192 * 128 /64
cfg["discount_factor"] = 0.995
cfg["lambda"] = 0.95 #0.95 0.97

def get_custom_optimizer(params, lr, **kwargs):
    policy_params = []
    attention_params = []
    std_params = []
    use_separate_attention_lr = bool(
        getattr(CONFIG, "use_separate_attention_lr", True)
    )

    # Keep the rapidly-changing fusion representation on a lower learning
    # rate while leaving the encoders, GRU, and policy/value heads unchanged.
    attention_parameter_roots = {
        "camera_proj",
        "map_proj",
        "pose_proj",
        "token_norm",
        "modality_embeddings",
        "cls_token",
        "sensor_attention",
        "mha",
        "mha_norm",
    }
    for name, p in models["policy"].named_parameters():
        if "log_std_parameter" in name:
            std_params.append(p)
        elif (
            use_separate_attention_lr
            and name.split(".", 1)[0] in attention_parameter_roots
        ):
            attention_params.append(p)
        else:
            policy_params.append(p)

    if (
        use_separate_attention_lr
        and getattr(CONFIG, "use_attention_fusion", False)
        and not attention_params
    ):
        raise RuntimeError(
            "Attention fusion is enabled, but no attention parameters were assigned "
            "to the reduced-learning-rate optimizer group"
        )
    
    opt_class = Muon if getattr(CONFIG, "optimizer_class", "adam").lower() == "muon" else torch.optim.Adam

    parameter_groups = [
        {"params": policy_params, "lr": lr, "name": "policy"},
    ]
    if attention_params:
        parameter_groups.append(
            {
                "params": attention_params,
                "lr": getattr(CONFIG, "attention_learning_rate", 1.5e-5),
                "name": "attention_fusion",
            }
        )

    if getattr(CONFIG, "manual_std_decay", False):
        print("[INFO] manual_std_decay is True. Removing log_std_parameter from optimizer.")
    else:
        parameter_groups.append(
            {
                "params": std_params,
                "lr": getattr(CONFIG, "std_learning_rate", 3e-4),
                "name": "action_std",
            }
        )

    group_summary = ", ".join(
        f"{group['name']}: lr={group['lr']:.2e}, params={sum(p.numel() for p in group['params']):,}"
        for group in parameter_groups
    )
    print(f"[INFO] Optimizer parameter groups: {group_summary}")
    return opt_class(parameter_groups, **kwargs)

print("[INFO] Using custom optimizer builder for policy, attention-fusion, and action-std learning rates")
cfg["optimizer_class"] = get_custom_optimizer

scheduler_max_steps = (total_timesteps // rollout_length) * cfg["learning_epochs"]


cfg["learning_rate"] = CONFIG.learning_rate
cfg["learning_rate_scheduler"] = CONFIG.scheduler_class
cfg["learning_rate_scheduler_kwargs"] = CONFIG.scheduler_kwargs.copy()

if "total_iters" in cfg["learning_rate_scheduler_kwargs"]:
        cfg["learning_rate_scheduler_kwargs"]["total_iters"] = scheduler_max_steps # Delayed decay # 50 mil steps we do 3/10
elif "T_max" in cfg["learning_rate_scheduler_kwargs"]:
        cfg["learning_rate_scheduler_kwargs"]["T_max"] = scheduler_max_steps
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = getattr(CONFIG, "grad_clip_norm", 0.7)
cfg["ratio_clip"] = 0.2
cfg["kl_threshold"] = getattr(CONFIG, "kl_threshold", 0.0)
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = CONFIG.entropy_coef
cfg["value_loss_scale"] = getattr(CONFIG, "value_loss_scale", 1.0)
cfg["action_std_names"] = ["linear_velocity", "angular_velocity", "pan", "tilt"]
cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * 1.0
cfg["time_limit_bootstrap"] = True

# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["state_preprocessor"] = None
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

script_dir = os.path.dirname(os.path.abspath(__file__))
log_root_path = os.path.join(script_dir, "logs", "skrl", "Alblation-Baseline")
log_root_path = os.path.abspath(log_root_path)

# experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_gru_128"
# experiment_name = "Buld_dataset_2"
# experiment_name = "SEEIR-Baseline-FT" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# experiment_name = "Pretrain" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
time_stmp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# experiment_name = "Alblation_MLP_FUSION" + "_"+time_stmp
experiment_name = "Alblation_ATTN_FUS" + "_"+time_stmp
log_dir = os.path.join(log_root_path, experiment_name)

is_main_process = int(os.environ.get("REAL_LOCAL_RANK", 0)) == 0
_use_wandb = _use_wandb and is_main_process

if is_eval:
    # Evaluation results are written next to the loaded checkpoint below. Do not
    # create a timestamped training run or initialize SKRL's training writers.
    _use_wandb = False
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["wandb"] = False
else:
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] skrl will log this experiment in: {log_dir}")
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

if not is_eval and _use_wandb:
    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["name"] = "IsaacLab-scripts_reinforcement_learning_skrl"
    cfg["experiment"]["checkpoint_interval"] = 3_000
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = experiment_name
    cfg["experiment"]["wandb"] = _use_wandb  # Disable wandb in evaluation mode

if _use_wandb:
    # Build a comprehensive config dictionary for WandB
    wandb_config = {}
    
    # 1. Log SKRL CONFIG
    if "CONFIG" in globals():
        wandb_config["SKRL_CONFIG"] = {k: v for k, v in CONFIG.__dict__.items() if not k.startswith('__') and not callable(v)}
        
    # 2. Log Environment config components (rewards, robot, mapping)
    if "env_cfg" in globals():
        if hasattr(env_cfg, "reward_cfg"):
            wandb_config["REWARDS_CFG"] = {k: v for k, v in env_cfg.reward_cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
        if hasattr(env_cfg, "robot_cfg"):
            wandb_config["ROBOT_CFG"] = {k: v for k, v in env_cfg.robot_cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
        if hasattr(env_cfg, "mapping_cfg"):
            wandb_config["MAPPING_CFG"] = {k: v for k, v in env_cfg.mapping_cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
        
    # 3. Log task run_config (from isaaclab_tasks - only logs the active cfg_mode)
    try:
        from isaaclab_tasks.direct.robot_inspection.run_config import cfg_mode as env_run_cfg
        wandb_config["RUN_CONFIG"] = {k: v for k, v in env_run_cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
    except Exception as e:
        print(f"[WARNING] Could not load RUN_CONFIG for WandB: {e}")

    cfg["experiment"]["wandb_kwargs"] = {
        "project": "Multi_object_inspection",  # Name of the project in WandB dashboard
        "name": experiment_name,           # Name of this specific run
        "tags": ["PPO", "IsaacLab", args_cli.task],
        "config": wandb_config
    }

    # Try to extract curriculum config if available
    try:
        # Access the base environment
        if hasattr(env, "unwrapped"):
             base_env = env.unwrapped
        else:
             base_env = env
        
        if hasattr(base_env, "curriculum"):
            curr = base_env.curriculum
            cfg["experiment"]["wandb_kwargs"]["config"].update({
                "curr_start_coverage": getattr(curr, "start_coverage_ratio", "N/A"),
                "curr_max_coverage": getattr(curr, "max_coverage_threshold", "N/A"),
                 # Check for both "treshold" (typo in file) and "threshold"
                "curr_quality_start": getattr(curr, "start_quality_threshold", getattr(curr, "start_quality_treshold", "N/A")),  
                "curr_quality_max": getattr(curr, "max_quality_threshold", getattr(curr, "max_quality_treshold", "N/A")),
                
                "curr_inc_up": getattr(curr, "coverage_increment_up", "N/A"),
                "curr_inc_down": getattr(curr, "coverage_increment_down", "N/A"),
                "curr_quality_inc": getattr(curr, "quality_increment", "N/A"),
                
                "curr_success_thresh_up": getattr(curr, "success_rate_increase_thresh", "N/A"),
                "curr_success_thresh_down": getattr(curr, "success_rate_decrease_thresh", "N/A"),
                
                "curr_min_ep_len": getattr(curr, "min_episode_length_limit", "N/A"),
                "curr_max_ep_len": getattr(curr, "max_episode_length_limit", "N/A"),
            })
            print(f"[INFO] Added Curriculum config to WandB: {cfg['experiment']['wandb_kwargs']['config']}")
        else:
            print("[WARNING] Could not find 'curriculum' in environment for WandB logging.")
            
    except Exception as e:
        print(f"[WARNING] Failed to extract curriculum config for WandB: {e}")
# Pass manual decay variables into agent's configuration for SKRL library hook
cfg["manual_std_decay"] = getattr(CONFIG, "manual_std_decay", False)
cfg["init_log_std"] = getattr(CONFIG, "init_log_std", 0.0)
cfg["final_log_std"] = getattr(CONFIG, "final_log_std", -2.0)
cfg["std_decay_fraction"] = getattr(CONFIG, "std_decay_fraction", 0.25)

agent = PPO(models=models, 
            memory=memory,
            cfg=cfg,
            observation_space = env.observation_space,
            action_space=env.action_space,
            device=env.device,)
# path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1862000.pt"
# agent.load(path)

if args_cli.checkpoint:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args_cli.checkpoint}")
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)
    if args_cli.reset_std:
        print("[INFO] Resetting log_std_parameter to force exploration.")
        with torch.no_grad():
             # Assuming shared model or separate policy has this specific parameter name
             if hasattr(agent.policy, "log_std_parameter"):
                 agent.policy.log_std_parameter.fill_(getattr(CONFIG, "init_log_std", 0.0)) # Reset to initial configured value
             else:
                 print("[WARNING] Could not find log_std_parameter to reset.")
elif is_eval:
    raise ValueError("Evaluation requires --checkpoint PATH (or an evaluation CONFIG with checkpoint_path set).")
cfg_trainer ={"timesteps": total_timesteps,  # total timesteps to train the agent
                "headless": _headless,
               }#  "stochastic_evaluation": False

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
print("[INFO] Starting evaluation..." if is_eval else "[INFO] Starting training...")
print_dict(cfg_trainer, nesting=4)
if is_eval: 
    print("[INFO] Running in evaluation mode. No training will be performed.")
    path = args_cli.checkpoint
    # Custom evaluation loop to handle RNN states
    print("[INFO] Starting custom evaluation loop with RNN state management...")
    
    # Initialize agent for evaluation
    agent.set_running_mode("eval")
    deterministic_eval = getattr(CONFIG, "deterministic_eval", True)
    evaluation_mode = "deterministic (policy mean)" if deterministic_eval else "stochastic (policy sample)"
    print(f"[INFO] Evaluation action mode: {evaluation_mode}")
    
    # Reset environment
    states, _ = env.reset()
    
    # Initialize RNN states if applicable
    if agent._rnn:
        # Reset internal RNN states
        for rnn_state in agent._rnn_initial_states["policy"]:
            rnn_state.zero_()
        if agent.policy is not agent.value:
            for rnn_state in agent._rnn_initial_states["value"]:
                rnn_state.zero_()

    episode_count = 0
    faces_discovered_list = []
    coverage_percent_list = []
    target_index_list = []
    crashes_list = []
    episode_steps_list = []
    episode_success_list = []
    episode_timed_out_list = []
    inspection_quality_list = []
    completed_env_id_list = []
    coverage_at_step_1000_list = []
    faces_at_step_1000_list = []
    crash_source_counts_list = []
    forward_crashes_list = []
    reverse_crashes_list = []
    safety_shield_interventions_list = []
    safety_shield_intervention_rate_list = []
    safety_shield_forward_interventions_list = []
    safety_shield_reverse_interventions_list = []
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    evaluation_num_envs = int(env.num_envs)
    target_index_to_name = tuple(getattr(base_env, "target_index_to_name", ()))
    crash_source_names = tuple(getattr(base_env, "crash_source_names", ()))
    evaluation_episode_horizon = int(
        base_env.curriculum.get_current_episode_length()
    )
    coverage_at_step_1000 = torch.full(
        (evaluation_num_envs,), float("nan"), device=env.device
    )
    faces_at_step_1000 = torch.full(
        (evaluation_num_envs,), -1, dtype=torch.long, device=env.device
    )
    eval_dir = os.path.join(os.path.dirname(path), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"[INFO] Evaluation results will be saved to: {eval_dir}")

    def _flat_values(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, np.ndarray):
            return value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return np.asarray(value).reshape(-1).tolist()
        return [value]

    def _row_values(value, row_width):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1, row_width).tolist()
        return np.asarray(value).reshape(-1, row_width).tolist()

    with torch.no_grad():
        try:
            while simulation_app.is_running():
                # Capture coverage when an episode first crosses step 1000. This
                # lets a 1200-step evaluation quantify how much was gained in
                # the final 200 steps without altering the frozen policy.
                if (
                    hasattr(base_env, "episode_length_buf")
                    and hasattr(base_env, "best_q_per_face")
                    and hasattr(base_env, "coverage_num_faces")
                ):
                    snapshot_mask = (
                        (base_env.episode_length_buf >= 1000)
                        & torch.isnan(coverage_at_step_1000)
                    )
                    snapshot_env_ids = snapshot_mask.nonzero(as_tuple=False).squeeze(-1)
                    if snapshot_env_ids.numel() > 0:
                        snapshot_q = base_env.best_q_per_face[snapshot_env_ids]
                        snapshot_faces = (snapshot_q > 0.0).sum(dim=1)
                        snapshot_denominator = base_env.coverage_num_faces[
                            snapshot_env_ids
                        ].float().clamp_min(1.0)
                        faces_at_step_1000[snapshot_env_ids] = snapshot_faces
                        coverage_at_step_1000[snapshot_env_ids] = (
                            snapshot_faces.float() / snapshot_denominator * 100.0
                        )

                # Get actions using the agent (handles RNN state internally via _rnn_initial_states)
                # The agent.act() method uses _rnn_initial_states, computes output, and populates _rnn_final_states
                actions, _, outputs = agent.act(states, timestep=0, timesteps=0)
                if deterministic_eval:
                    actions = outputs["mean_actions"]

                # Step environment
                next_states, rewards, terminated, truncated, infos = env.step(actions)

                finished_mask = terminated | truncated
                if torch.any(finished_mask):
                    finished_env_ids = finished_mask.nonzero(as_tuple=False)[:, 0]
                    completed_this_step = int(finished_env_ids.numel())
                    remaining = completed_this_step
                    if args_cli.max_episodes is not None:
                        remaining = min(
                            remaining,
                            max(0, args_cli.max_episodes - len(faces_discovered_list)),
                        )

                    # Collect stats
                    if remaining > 0 and "log" in infos:
                        if "faces_discovered" in infos["log"]:
                            log_values = infos["log"]
                            val = log_values["faces_discovered"]
                            coverage_percent = log_values.get("coverage_percent", None)
                            target_indices = log_values.get("target_index", None)
                            crashes = log_values.get("crashes", None)
                            episode_steps = log_values.get("episode_steps", None)
                            episode_success = log_values.get("episode_success", None)
                            episode_timed_out = log_values.get("episode_timed_out", None)
                            inspection_quality = log_values.get(
                                "mean_inspection_quality", None
                            )
                            crash_source_counts = log_values.get("crash_source_counts", None)
                            forward_crashes = log_values.get("forward_crashes", None)
                            reverse_crashes = log_values.get("reverse_crashes", None)
                            shield_interventions = infos["log"].get(
                                "safety_shield_interventions", None
                            )
                            shield_intervention_rate = infos["log"].get(
                                "safety_shield_intervention_rate", None
                            )
                            shield_forward_interventions = infos["log"].get(
                                "safety_shield_forward_interventions", None
                            )
                            shield_reverse_interventions = infos["log"].get(
                                "safety_shield_reverse_interventions", None
                            )

                            faces_values = _flat_values(val)[:remaining]
                            coverage_values = _flat_values(coverage_percent)[:remaining]
                            crash_values = _flat_values(crashes)[:remaining]
                            step_values = _flat_values(episode_steps)[:remaining]
                            source_rows = _row_values(
                                crash_source_counts, len(crash_source_names)
                            )[:remaining] if crash_source_names else []

                            faces_discovered_list.extend(faces_values)
                            coverage_percent_list.extend(coverage_values)
                            target_index_list.extend(
                                _flat_values(target_indices)[:remaining]
                            )
                            crashes_list.extend(crash_values)
                            episode_steps_list.extend(step_values)
                            episode_success_list.extend(
                                _flat_values(episode_success)[:remaining]
                            )
                            episode_timed_out_list.extend(
                                _flat_values(episode_timed_out)[:remaining]
                            )
                            inspection_quality_list.extend(
                                _flat_values(inspection_quality)[:remaining]
                            )
                            crash_source_counts_list.extend(source_rows)
                            forward_crashes_list.extend(
                                _flat_values(forward_crashes)[:remaining]
                            )
                            reverse_crashes_list.extend(
                                _flat_values(reverse_crashes)[:remaining]
                            )

                            kept_env_ids = finished_env_ids[:remaining]
                            completed_env_id_list.extend(
                                kept_env_ids.detach().cpu().tolist()
                            )
                            snapshot_coverages = coverage_at_step_1000[
                                kept_env_ids
                            ].detach().cpu().tolist()
                            snapshot_faces = faces_at_step_1000[
                                kept_env_ids
                            ].detach().cpu().tolist()
                            for index in range(remaining):
                                # A 1000-step evaluation currently truncates at
                                # recorded step 999. Treat that final value as
                                # the step-1000 boundary for horizon reporting.
                                if (
                                    np.isnan(snapshot_coverages[index])
                                    and index < len(step_values)
                                    and step_values[index] >= 999
                                    and index < len(coverage_values)
                                ):
                                    snapshot_coverages[index] = coverage_values[index]
                                    snapshot_faces[index] = faces_values[index]
                            coverage_at_step_1000_list.extend(snapshot_coverages)
                            faces_at_step_1000_list.extend(snapshot_faces)

                            for value, destination in (
                                (shield_interventions, safety_shield_interventions_list),
                                (shield_intervention_rate, safety_shield_intervention_rate_list),
                                (shield_forward_interventions, safety_shield_forward_interventions_list),
                                (shield_reverse_interventions, safety_shield_reverse_interventions_list),
                            ):
                                if value is None:
                                    continue
                                destination.extend(_flat_values(value)[:remaining])

                            first_episode_number = len(faces_discovered_list) - remaining + 1
                            for index, face_count in enumerate(faces_values):
                                crash_str = ""
                                if index < len(crash_values) and crash_values[index]:
                                    if index < len(source_rows):
                                        source_index = int(np.argmax(source_rows[index]))
                                        source = crash_source_names[source_index]
                                        if source.startswith("obstacle_"):
                                            source = "obstacle"
                                        crash_str = f" | Crash: {source}"
                                step_str = (
                                    f" | Steps: {int(step_values[index])}"
                                    if index < len(step_values)
                                    else ""
                                )
                                print(
                                    f"[INFO] Completion {first_episode_number + index} "
                                    f"(env {int(kept_env_ids[index])}) "
                                    f"Faces Discovered: {face_count}{step_str}{crash_str}"
                                )

                    coverage_at_step_1000[finished_env_ids] = float("nan")
                    faces_at_step_1000[finished_env_ids] = -1
                    episode_count = len(faces_discovered_list)
                    if args_cli.max_episodes is not None and episode_count >= args_cli.max_episodes:
                        print(f"[INFO] strict max_episodes reached: {episode_count}")
                        break
                
                # Update RNN states for next step
                if agent._rnn:
                    # Move final states to initial states for next step
                    # Note: agent._rnn_final_states is updated inside agent.act()
                    agent._rnn_initial_states = agent._rnn_final_states
                    
                    # Reset RNN states for terminated episodes
                    # The agent.record_transition method usually does this, but we are skipping it in eval
                    # so we must do it manually
                    finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
                    if finished_episodes.numel():
                        for rnn_state in agent._rnn_initial_states["policy"]:
                            rnn_state[:, finished_episodes[:, 0]] = 0
                        if agent.policy is not agent.value:
                            for rnn_state in agent._rnn_initial_states["value"]:
                                rnn_state[:, finished_episodes[:, 0]] = 0

                # Update current state
                states = next_states
        except KeyboardInterrupt:
            print("[INFO] Keyboard interrupt detected. Exiting evaluation loop early.")
        finally:
            print("[INFO] Closing environment, ensuring data is saved...")
            env.close()

    # Print Final Statistics
    if len(faces_discovered_list) > 0:
        faces_array = np.array(faces_discovered_list)
        summary = {
            "checkpoint": path,
            "evaluation_mode": "deterministic" if deterministic_eval else "stochastic",
            "seed": WORKER_SEED,
            "num_envs": evaluation_num_envs,
            "configured_episode_horizon": evaluation_episode_horizon,
            "episodes": int(len(faces_array)),
            "faces": {
                "mean": float(np.mean(faces_array)),
                "std": float(np.std(faces_array)),
                "median": float(np.median(faces_array)),
                "min": int(np.min(faces_array)),
                "max": int(np.max(faces_array)),
                "p01": float(np.percentile(faces_array, 1)),
                "p05": float(np.percentile(faces_array, 5)),
                "p95": float(np.percentile(faces_array, 95)),
            },
        }
        coverage_array = np.asarray(coverage_percent_list, dtype=np.float64)
        target_index_array = np.asarray(target_index_list, dtype=np.int64)
        episode_steps_array = np.asarray(episode_steps_list, dtype=np.int64)
        episode_success_array = np.asarray(episode_success_list, dtype=np.bool_)
        episode_timed_out_array = np.asarray(episode_timed_out_list, dtype=np.bool_)
        inspection_quality_array = np.asarray(
            inspection_quality_list, dtype=np.float64
        )
        completed_env_id_array = np.asarray(completed_env_id_list, dtype=np.int64)
        coverage_at_step_1000_array = np.asarray(
            coverage_at_step_1000_list, dtype=np.float64
        )
        faces_at_step_1000_array = np.asarray(
            faces_at_step_1000_list, dtype=np.int64
        )
        shield_interventions_array = np.asarray(
            safety_shield_interventions_list, dtype=np.int64
        )
        shield_intervention_rate_array = np.asarray(
            safety_shield_intervention_rate_list, dtype=np.float64
        )
        shield_forward_array = np.asarray(
            safety_shield_forward_interventions_list, dtype=np.int64
        )
        shield_reverse_array = np.asarray(
            safety_shield_reverse_interventions_list, dtype=np.int64
        )
        if len(coverage_array) == len(faces_array):
            summary["coverage_percent"] = {
                "mean": float(np.mean(coverage_array)),
                "std": float(np.std(coverage_array)),
                "median": float(np.median(coverage_array)),
                "min": float(np.min(coverage_array)),
                "max": float(np.max(coverage_array)),
            }

        if len(inspection_quality_array) == len(faces_array):
            summary["inspection_quality"] = {
                "mean": float(np.mean(inspection_quality_array)),
                "std": float(np.std(inspection_quality_array)),
                "median": float(np.median(inspection_quality_array)),
                "p90": float(np.percentile(inspection_quality_array, 90)),
                "p95": float(np.percentile(inspection_quality_array, 95)),
            }

        if len(episode_steps_array) == len(faces_array):
            summary["episode_length_steps"] = {
                "mean": float(np.mean(episode_steps_array)),
                "median": float(np.median(episode_steps_array)),
                "p90": float(np.percentile(episode_steps_array, 90)),
                "p95": float(np.percentile(episode_steps_array, 95)),
                "min": int(np.min(episode_steps_array)),
                "max": int(np.max(episode_steps_array)),
            }
            summary["episode_horizon_milestones"] = {}
            for milestone in (1000, 1200):
                # _get_dones currently truncates at max_steps - 1, so a
                # configured 1000-step horizon is recorded as step 999.
                reached_mask = episode_steps_array >= milestone - 1
                milestone_summary = {
                    "recorded_step_cutoff": milestone - 1,
                    "episodes_reaching": int(np.sum(reached_mask)),
                    "episodes_reaching_percent": float(
                        np.mean(reached_mask) * 100.0
                    ),
                }
                if np.any(reached_mask) and len(coverage_array) == len(faces_array):
                    milestone_summary["mean_final_coverage_percent"] = float(
                        np.mean(coverage_array[reached_mask])
                    )
                if np.any(reached_mask) and len(crashes_list) == len(faces_array):
                    reached_crashes = np.asarray(crashes_list)[reached_mask] > 0
                    milestone_summary["crashes"] = int(np.sum(reached_crashes))
                    milestone_summary["crash_rate_percent"] = float(
                        np.mean(reached_crashes) * 100.0
                    )
                summary["episode_horizon_milestones"][str(milestone)] = (
                    milestone_summary
                )

        if len(episode_success_array) == len(faces_array):
            summary["success"] = {
                "episodes": int(np.sum(episode_success_array)),
                "rate_percent": float(np.mean(episode_success_array) * 100.0),
            }
        if len(episode_timed_out_array) == len(faces_array):
            summary["timeouts"] = {
                "episodes": int(np.sum(episode_timed_out_array)),
                "rate_percent": float(np.mean(episode_timed_out_array) * 100.0),
            }

        if (
            len(coverage_at_step_1000_array) == len(faces_array)
            and len(coverage_array) == len(faces_array)
        ):
            reached_1000_mask = ~np.isnan(coverage_at_step_1000_array)
            after_1000_summary = {
                "episodes_reaching_step_1000": int(np.sum(reached_1000_mask)),
                "episodes_reaching_step_1000_percent": float(
                    np.mean(reached_1000_mask) * 100.0
                ),
            }
            if np.any(reached_1000_mask):
                coverage_gain = (
                    coverage_array[reached_1000_mask]
                    - coverage_at_step_1000_array[reached_1000_mask]
                )
                after_1000_summary.update(
                    {
                        "mean_coverage_at_step_1000_percent": float(
                            np.mean(coverage_at_step_1000_array[reached_1000_mask])
                        ),
                        "mean_final_coverage_percent": float(
                            np.mean(coverage_array[reached_1000_mask])
                        ),
                        "mean_coverage_gained_after_step_1000_percent": float(
                            np.mean(coverage_gain)
                        ),
                    }
                )
            if (
                len(episode_steps_array) == len(faces_array)
                and len(crashes_list) == len(faces_array)
            ):
                crash_after_1000_mask = (
                    (episode_steps_array > 1000)
                    & (np.asarray(crashes_list) > 0)
                )
                after_1000_summary["crashes_after_step_1000"] = int(
                    np.sum(crash_after_1000_mask)
                )
            summary["after_step_1000"] = after_1000_summary

        if len(target_index_array) == len(faces_array) and target_index_to_name:
            per_target = {}
            for target_index, target_name in enumerate(target_index_to_name):
                target_mask = target_index_array == target_index
                if not np.any(target_mask):
                    continue
                target_faces = faces_array[target_mask]
                target_summary = {
                    "episodes": int(np.sum(target_mask)),
                    "mean_faces": float(np.mean(target_faces)),
                }
                if len(coverage_array) == len(faces_array):
                    target_summary["mean_coverage_percent"] = float(
                        np.mean(coverage_array[target_mask])
                    )
                if len(inspection_quality_array) == len(faces_array):
                    target_summary["mean_inspection_quality"] = float(
                        np.mean(inspection_quality_array[target_mask])
                    )
                per_target[target_name] = target_summary
            summary["per_target"] = per_target
        print("\n" + "="*50)
        print(f"EVALUATION RESULTS ({len(faces_discovered_list)} Episodes)")
        print("="*50)
        print(f"Mean Faces Discovered: {np.mean(faces_array):.2f}")
        print(f"Std Deviation:         {np.std(faces_array):.2f}")
        print(f"Min Faces:             {np.min(faces_array)}")
        print(f"Max Faces:             {np.max(faces_array)}")
        if len(coverage_array) == len(faces_array):
            print(f"Mean Coverage:          {np.mean(coverage_array):.2f}%")
            print(f"Std Coverage:           {np.std(coverage_array):.2f}%")
        if len(inspection_quality_array) == len(faces_array):
            print(f"Mean Inspection Quality: {np.mean(inspection_quality_array):.4f}")
        if len(episode_steps_array) == len(faces_array):
            print("Episode Lengths:")
            print(f"  Median: {np.median(episode_steps_array):.1f} steps")
            print(f"  P90:    {np.percentile(episode_steps_array, 90):.1f} steps")
            print(f"  P95:    {np.percentile(episode_steps_array, 95):.1f} steps")
            for milestone in (1000, 1200):
                milestone_result = summary["episode_horizon_milestones"][str(milestone)]
                print(
                    f"  Reached {milestone}-step horizon: "
                    f"{milestone_result['episodes_reaching']} "
                    f"({milestone_result['episodes_reaching_percent']:.2f}%)"
                )
        if "success" in summary:
            print(
                f"Success Rate:            {summary['success']['rate_percent']:.2f}% "
                f"({summary['success']['episodes']} episodes)"
            )
        if "timeouts" in summary:
            print(
                f"Timeout Rate:            {summary['timeouts']['rate_percent']:.2f}% "
                f"({summary['timeouts']['episodes']} episodes)"
            )
        if "after_step_1000" in summary:
            late_summary = summary["after_step_1000"]
            print("After Step 1000:")
            print(
                "  Episodes reaching boundary: "
                f"{late_summary['episodes_reaching_step_1000']} "
                f"({late_summary['episodes_reaching_step_1000_percent']:.2f}%)"
            )
            if "mean_coverage_at_step_1000_percent" in late_summary:
                print(
                    "  Mean coverage at step 1000: "
                    f"{late_summary['mean_coverage_at_step_1000_percent']:.2f}%"
                )
                print(
                    "  Mean additional coverage:  "
                    f"{late_summary['mean_coverage_gained_after_step_1000_percent']:.2f}%"
                )
            if "crashes_after_step_1000" in late_summary:
                print(
                    "  Crashes after step 1000:    "
                    f"{late_summary['crashes_after_step_1000']}"
                )
        if "per_target" in summary:
            print("Per-target results:")
            for target_name, target_summary in summary["per_target"].items():
                coverage_text = ""
                if "mean_coverage_percent" in target_summary:
                    coverage_text = f" | Coverage: {target_summary['mean_coverage_percent']:.2f}%"
                print(
                    f"  {target_name}: {target_summary['episodes']} episodes"
                    f" | Faces: {target_summary['mean_faces']:.2f}{coverage_text}"
                )
        if crashes_list:
            crashes_array = np.asarray(crashes_list)
            total_terminated = int(np.sum(crashes_array > 0))
            
            # Calculate coverage only for successful episodes
            successful_episodes_mask = crashes_array == 0
            if np.any(successful_episodes_mask):
                faces_successful = faces_array[successful_episodes_mask]
                mean_faces_successful = float(np.mean(faces_successful))
            else:
                mean_faces_successful = 0.0

            summary["crashes"] = {
                "mean": float(np.mean(crashes_array)),
                "median": float(np.median(crashes_array)),
                "episodes_with_crash_percent": float(
                    np.mean(crashes_array > 0) * 100
                ),
                "total_terminated_due_to_crash": total_terminated,
            }
            summary["faces"]["mean_successful_only"] = mean_faces_successful
            
            print(f"Mean Crashes: {np.mean(crashes_array):.2f}")
            print(f"Median Crashes: {np.median(crashes_array):.2f}")
            print(
                "Episodes With Crash: "
                f"{np.mean(crashes_array > 0) * 100:.2f}%"
            )
            print(f"Total Terminated Due to Crash: {total_terminated} / {len(crashes_array)}")
            print(f"Mean Faces Discovered (NO CRASHES): {mean_faces_successful:.2f}")

            crash_source_counts_array = np.asarray(crash_source_counts_list, dtype=np.int64)
            if (
                crash_source_names
                and crash_source_counts_array.ndim == 2
                and crash_source_counts_array.shape == (len(faces_array), len(crash_source_names))
            ):
                source_totals = crash_source_counts_array.sum(axis=0)
                
                grouped_totals = {}
                for source_name, source_count in zip(crash_source_names, source_totals):
                    if source_name.startswith("obstacle_"):
                        group_name = "obstacle"
                    else:
                        group_name = source_name
                    grouped_totals[group_name] = grouped_totals.get(group_name, 0) + source_count
                    
                summary["crash_sources"] = {
                    source_name: {
                        "count": int(source_count),
                        "percent_of_crashes": float(
                            100.0 * source_count / max(1, source_totals.sum())
                        ),
                    }
                    for source_name, source_count in grouped_totals.items()
                }
                print("Crash Sources (primary base_link contact):")
                for source_name, source_count in grouped_totals.items():
                    print(
                        f"  {source_name}: {int(source_count)} "
                        f"({100.0 * source_count / max(1, source_totals.sum()):.2f}% of crashes)"
                    )

            forward_array = np.asarray(forward_crashes_list, dtype=np.int64)
            reverse_array = np.asarray(reverse_crashes_list, dtype=np.int64)
            if len(forward_array) > 0 and len(reverse_array) > 0:
                total_forward = forward_array.sum()
                total_reverse = reverse_array.sum()
                total_directional = max(1, total_forward + total_reverse)
                summary["crash_directions"] = {
                    "forward": {
                        "count": int(total_forward),
                        "percent_of_directional_crashes": float(100.0 * total_forward / total_directional)
                    },
                    "reverse": {
                        "count": int(total_reverse),
                        "percent_of_directional_crashes": float(100.0 * total_reverse / total_directional)
                    }
                }
                print("\nCrash Directions (intended velocity at time of crash):")
                print(f"  Forward: {int(total_forward)} ({100.0 * total_forward / total_directional:.2f}%)")
                print(f"  Reverse: {int(total_reverse)} ({100.0 * total_reverse / total_directional:.2f}%)")

        if len(shield_interventions_array) == len(faces_array):
            total_shield_interventions = int(shield_interventions_array.sum())
            shield_summary = {
                "mean_interventions_per_episode": float(np.mean(shield_interventions_array)),
                "episodes_with_intervention_percent": float(
                    np.mean(shield_interventions_array > 0) * 100.0
                ),
                "total_interventions": total_shield_interventions,
            }
            if len(shield_intervention_rate_array) == len(faces_array):
                shield_summary["mean_intervened_steps_percent"] = float(
                    np.mean(shield_intervention_rate_array) * 100.0
                )
            if (
                len(shield_forward_array) == len(faces_array)
                and len(shield_reverse_array) == len(faces_array)
            ):
                shield_summary["forward_interventions"] = int(shield_forward_array.sum())
                shield_summary["reverse_interventions"] = int(shield_reverse_array.sum())
            summary["safety_shield"] = shield_summary

            print("\nSafety Shield:")
            print(f"  Total interventions: {total_shield_interventions}")
            print(
                "  Episodes with intervention: "
                f"{np.mean(shield_interventions_array > 0) * 100.0:.2f}%"
            )
            if len(shield_intervention_rate_array) == len(faces_array):
                print(
                    "  Mean intervened steps: "
                    f"{np.mean(shield_intervention_rate_array) * 100.0:.2f}%"
                )

        print("="*50 + "\n")

        result_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_mode = "deterministic" if deterministic_eval else "stochastic"
        result_stem = f"eval_{result_mode}_{result_timestamp}"
        summary_path = os.path.join(eval_dir, f"{result_stem}.json")
        raw_path = os.path.join(eval_dir, f"{result_stem}.npz")

        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

        raw_results = {"faces_discovered": faces_array}
        if len(coverage_array) == len(faces_array):
            raw_results["coverage_percent"] = coverage_array
        if len(inspection_quality_array) == len(faces_array):
            raw_results["mean_inspection_quality"] = inspection_quality_array
        if len(episode_steps_array) == len(faces_array):
            raw_results["episode_steps"] = episode_steps_array
        if len(episode_success_array) == len(faces_array):
            raw_results["episode_success"] = episode_success_array
        if len(episode_timed_out_array) == len(faces_array):
            raw_results["episode_timed_out"] = episode_timed_out_array
        if len(completed_env_id_array) == len(faces_array):
            raw_results["completed_env_id"] = completed_env_id_array
        if len(coverage_at_step_1000_array) == len(faces_array):
            raw_results["coverage_at_step_1000_percent"] = (
                coverage_at_step_1000_array
            )
        if len(faces_at_step_1000_array) == len(faces_array):
            raw_results["faces_at_step_1000"] = faces_at_step_1000_array
        if len(target_index_array) == len(faces_array):
            raw_results["target_index"] = target_index_array
            raw_results["target_names"] = np.asarray(target_index_to_name)
        if crashes_list:
            raw_results["crashes"] = crashes_array
        if (
            crash_source_counts_list
            and crash_source_counts_array.shape == (len(faces_array), len(crash_source_names))
        ):
            raw_results["crash_source_counts"] = crash_source_counts_array
            raw_results["crash_source_names"] = np.asarray(crash_source_names)
            
        if forward_crashes_list:
            raw_results["forward_crashes"] = forward_array
        if reverse_crashes_list:
            raw_results["reverse_crashes"] = reverse_array
        if len(shield_interventions_array) == len(faces_array):
            raw_results["safety_shield_interventions"] = shield_interventions_array
        if len(shield_intervention_rate_array) == len(faces_array):
            raw_results["safety_shield_intervention_rate"] = shield_intervention_rate_array
        if len(shield_forward_array) == len(faces_array):
            raw_results["safety_shield_forward_interventions"] = shield_forward_array
        if len(shield_reverse_array) == len(faces_array):
            raw_results["safety_shield_reverse_interventions"] = shield_reverse_array
            
        np.savez_compressed(raw_path, **raw_results)

        print(f"[INFO] Evaluation summary saved to: {summary_path}")
        print(f"[INFO] Raw episode results saved to: {raw_path}")
    else:
        print("[WARNING] No episodes completed to calculate statistics.")

else:
    # path = "/home/tosin/IsaacLab_inspection/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2025-09-07_11-42-53_ppo_gru_128/checkpoints/agent_450000.pt"
    # agent.load(path)
    try:
        trainer.train()
    finally:
        env.close()

# Close the Kit application explicitly after the environment. Without this,
# Python's implicit plugin teardown can unload Replicator and PhysX out of
# order, producing shutdown warnings and occasionally hanging the process.
simulation_app.close()
