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
parser.add_argument("--max_episodes", type=int, default=20, help="Maximum number of episodes to run in evaluation mode.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
_use_wandb = CONFIG.use_wandb
_headless = CONFIG.headless
args_cli = parser.parse_args()
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
# set_seed(42)
set_seed(42, deterministic=True)
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
            self.d_model = 256  # Hidden dimension for tokens
            
            # Projectors
            self.camera_proj = nn.Linear(camera_features_size, self.d_model)
            self.map_proj = nn.Linear(map_features_size, self.d_model)
            self.pose_proj = nn.Linear(self.encoded_pose_dim, self.d_model)
            
            # Token Normalization (avoids clamping and stabilizes attention)
            self.token_norm = nn.LayerNorm(self.d_model)
            
            # Modality embeddings
            # self.modality_embeddings = nn.Parameter(torch.randn(1, 3, self.d_model))
            self.modality_embeddings = nn.Parameter(torch.randn(1, 3, self.d_model) * 0.02)
            
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
            
            # Output is merged attended tokens via mean pooling
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
            
            # Mean pool tokens along sequence dim: [batch_size, d_model]
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

env_cfg.seed = 42
env = gym.make(args_cli.task, cfg=env_cfg)
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
cfg["learning_epochs"] = 2  #8
cfg["mini_batches"] = 8   # 16 horizon_length * num_actors / minibatch_size   8192 * 128 /64
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.97 #0.95 0.97

def get_custom_optimizer(params, lr, **kwargs):
    policy_params = []
    std_params = []
    for name, p in models["policy"].named_parameters():
        if "log_std_parameter" in name:
            std_params.append(p)
        else:
            policy_params.append(p)
    
    opt_class = Muon if getattr(CONFIG, "optimizer_class", "adam").lower() == "muon" else torch.optim.Adam
    
    if getattr(CONFIG, "manual_std_decay", False):
        print("[INFO] manual_std_decay is True. Removing log_std_parameter from optimizer.")
        return opt_class([
            {"params": policy_params, "lr": lr}
        ], **kwargs)
    else:
        return opt_class([
            {"params": policy_params, "lr": lr},
            {"params": std_params, "lr": getattr(CONFIG, "std_learning_rate", 3e-4)}
        ], **kwargs)

print("[INFO] Using custom optimizer builder to decouple log_std learning rate")
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
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = CONFIG.entropy_coef
cfg["value_loss_scale"] = getattr(CONFIG, "value_loss_scale", 1.0)
cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * 1.0
cfg["time_limit_bootstrap"] = True

# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["state_preprocessor"] = None
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

script_dir = os.path.dirname(os.path.abspath(__file__))
log_root_path = os.path.join(script_dir, "logs", "skrl", "SEEIR-Baseline")
log_root_path = os.path.abspath(log_root_path)

# experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_gru_128"
# experiment_name = "Buld_dataset_2"
# experiment_name = "SEEIR-Baseline-FT" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# experiment_name = "Pretrain" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "SEEIR-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    cfg["experiment"]["wandb_kwargs"] = {
        "project": "Multi_object_inspection",  # Name of the project in WandB dashboard
        "name": experiment_name,           # Name of this specific run
        "tags": ["PPO", "IsaacLab", args_cli.task],
        # "config": {}
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
cfg_trainer ={"timesteps": total_timesteps,  # total timesteps to train the agent
                "headless": _headless,
               }#  "stochastic_evaluation": False

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
print("[INFO] Starting training...")
print_dict(cfg_trainer, nesting=4)
if is_eval: 
    print("[INFO] Running in evaluation mode. No training will be performed.")
    if CONFIG.checkpoint_path:
        path = CONFIG.checkpoint_path
    else:
         # Fallback or error if you prefer, but aiming for minimal disruption
         path = "/home/tosin/logs/skrl/3DInspection_direct/2026-01-18_18-56-03_ppo_gru_128/checkpoints/agent_155000.pt" 
         print(f"[WARNING] No checkpoint path in CONFIG, using default: {path}")

    agent.load(path)
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
    crashes_list = []
    eval_dir = os.path.join(os.path.dirname(CONFIG.checkpoint_path), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"[INFO] Evaluation results will be saved to: {eval_dir}")

    with torch.no_grad():
        try:
            while simulation_app.is_running():
                # Get actions using the agent (handles RNN state internally via _rnn_initial_states)
                # The agent.act() method uses _rnn_initial_states, computes output, and populates _rnn_final_states
                actions, _, outputs = agent.act(states, timestep=0, timesteps=0)
                if deterministic_eval:
                    actions = outputs["mean_actions"]

                # Step environment
                next_states, rewards, terminated, truncated, infos = env.step(actions)

                if torch.any(terminated | truncated):
                    episode_count += torch.sum(terminated | truncated).item()
                    
                    # Collect stats
                    if "log" in infos:
                        if "faces_discovered" in infos["log"]:
                            # Assuming infos["log"]["faces_discovered"] is a tensor matching num_envs or similar
                            # We need to extract the value for the terminated env(s)
                            # For single env eval, it's straightforward.
                            val = infos["log"]["faces_discovered"]
                            val_dist = infos["log"].get("max_distance", None)
                            crashes = infos["log"].get("crashes", None)

                            if crashes is not None:
                                if isinstance(crashes, torch.Tensor):
                                    crashes_list.extend(
                                        crashes.detach().cpu().reshape(-1).tolist()
                                    )
                                else:
                                    crashes_list.append(crashes)

                            if isinstance(val, torch.Tensor):
                                 if val.numel() > 1:
                                     faces_discovered_list.extend(val.tolist())
                                     current_faces = val.max().item() # Approximate best in batch
                                     # Print summary for batch if desired, or skip to avoid spam
                                     print(f"[INFO] Batch of {val.numel()} episodes finished. Faces: {val.tolist()}")
                                 else:
                                     faces_discovered_list.append(val.item())
                                     current_faces = val.item()
                                     
                                     dist_str = ""
                                     if val_dist is not None:
                                         d_val = val_dist.item() if isinstance(val_dist, torch.Tensor) else val_dist
                                         dist_str = f" | Max Distance: {d_val:.2f}"
                                     print(f"[INFO] Episode {episode_count} Faces Discovered: {val.item()}{dist_str}")
                            else:
                                 faces_discovered_list.append(val)
                                 current_faces = val
                                 
                                 dist_str = ""
                                 if val_dist is not None:
                                     dist_str = f" | Max Distance: {val_dist:.2f}"
                                 print(f"[INFO] Episode {episode_count} Faces Discovered: {val}{dist_str}")

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
            "checkpoint": CONFIG.checkpoint_path,
            "evaluation_mode": "deterministic" if deterministic_eval else "stochastic",
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
        print("\n" + "="*50)
        print(f"EVALUATION RESULTS ({len(faces_discovered_list)} Episodes)")
        print("="*50)
        print(f"Mean Faces Discovered: {np.mean(faces_array):.2f}")
        print(f"Std Deviation:         {np.std(faces_array):.2f}")
        print(f"Min Faces:             {np.min(faces_array)}")
        print(f"Max Faces:             {np.max(faces_array)}")
        if crashes_list:
            crashes_array = np.asarray(crashes_list)
            summary["crashes"] = {
                "mean": float(np.mean(crashes_array)),
                "median": float(np.median(crashes_array)),
                "episodes_with_crash_percent": float(
                    np.mean(crashes_array > 0) * 100
                ),
            }
            print(f"Mean Crashes: {np.mean(crashes_array):.2f}")
            print(f"Median Crashes: {np.median(crashes_array):.2f}")
            print(
                "Episodes With Crash: "
                f"{np.mean(crashes_array > 0) * 100:.2f}%"
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
        if crashes_list:
            raw_results["crashes"] = crashes_array
        np.savez_compressed(raw_path, **raw_results)

        print(f"[INFO] Evaluation summary saved to: {summary_path}")
        print(f"[INFO] Raw episode results saved to: {raw_path}")
    else:
        print("[WARNING] No episodes completed to calculate statistics.")

else:
    # path = "/home/tosin/IsaacLab_inspection/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2025-09-07_11-42-53_ppo_gru_128/checkpoints/agent_450000.pt"
    # agent.load(path)
    trainer.train()
