import argparse
import os
import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import warnings
from heavyball import ForeachMuon
import heavyball.utils
torch.autograd.set_detect_anomaly(True)

# conda install -c conda-forge gcc=12 -y
from isaaclab.app import AppLauncher
is_eval = True
# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-Direct-v0", help="Name of the task.")
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
_use_wandb = False
_headless = False
args_cli = parser.parse_args()
args_cli.enable_cameras =  True
args_cli.headless = _headless
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
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
# sys.argv.append("--headless")
sys.argv.append("--enable_cameras")

set_seed(42)
# for some reason changing the clip actionsvarialbe to true in thr training script causes this error
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                observation_space,
                action_space,
                device,
                cfg,
                clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2,
                num_envs=1,
                sequence_length=32,
                _hidden_size=128,
                _hidden_size_gru=256):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        DeterministicMixin.__init__(self, False)
        self.cfg = cfg
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self._hidden_size = _hidden_size
        self._hidden_size_gru = _hidden_size_gru

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
        self.gru_input_size = camera_features_size + self.robot_pose_dim + map_features_size
        self.gru_hidden_size = 512 #H output size of GRU
        self.gru_num_layers = 1
        # print(f"DEBUG: gru_input_size: {self.gru_input_size}")

        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)
        #output heads

        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions ),
            nn.Tanh()  
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        # Action Head, MU and STD
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))


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
        # camera_obs = states["cameras"].permute(0, 3, 1, 2)  # (batch, channels, height, width)

        camera_features = self.camera_encoder(camera_obs_permuted)
        map_features = self.map_encoder(local_map_permuted)
        
        combined_features = torch.cat((camera_features, map_features, robot_pose), dim=1)

        if self.training:
            # just return dummy action to debug sim
            # return torch.zeros((self.num_envs, self.num_actions), device=self.device), {"rnn": [hidden_states]}
            rnn_input = combined_features.view(-1, self.sequence_length, combined_features.shape[-1])
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
                    hidden_states[:, terminated[:, i1 - 1], :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            rnn_input = combined_features.unsqueeze(1)
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




env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env_cfg.seed = 42
env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)

device = env.device
# assume num env is 16
TOTAL_BATCH_SIZE = 8192 #8192# 2048
sequence_length = 32
rollout_length = TOTAL_BATCH_SIZE // env.num_envs

memory = RandomMemory(memory_size=rollout_length, num_envs=env.num_envs, device=device)
model_config = {
        "nonlinearity": "elu",
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
                            sequence_length=sequence_length)
models['value'] = models["policy"]  # Shared(env.observation_space, env.action_space, env.device)
total_timesteps = 100_000

cfg = PPO_DEFAULT_CONFIG.copy()
warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
heavyball.utils.compile_mode = None
cfg["rollouts"] = rollout_length  # memory_size
cfg["learning_epochs"] = 4 #8
cfg["mini_batches"] = 32  # horizon_length * num_actors / minibatch_size  : 4096 * 16
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-5  # 0.0003     0.0006
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016} # 0.008
scheduler_max_steps = (total_timesteps // rollout_length) * cfg["learning_epochs"]
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0.7
cfg["ratio_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 3e-4
cfg["value_loss_scale"] = 1.0
# cfg["kl_threshold"] = 0.0
cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * 0.1
cfg["rewards_shaper"] =  None
cfg["time_limit_bootstrap"] = True

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)

script_dir = os.path.dirname(os.path.abspath(__file__))
log_root_path = os.path.join(script_dir, "logs", "skrl", "3DInspection_direct")
log_root_path = os.path.abspath(log_root_path)

experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_gru_128"

print(f"[INFO] Logging experiment in directory: {log_root_path}")

log_dir = os.path.join(log_root_path, experiment_name)
print(f"[INFO] skrl will log this experiment in: {log_dir}")

os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)



cfg["experiment"]["write_interval"] = 2400
cfg["experiment"]["name"] = "IsaacLab-scripts_reinforcement_learning_skrl"
cfg["experiment"]["checkpoint_interval"] = 5_000
cfg["experiment"]["directory"] = log_root_path
cfg["experiment"]["experiment_name"] = experiment_name
cfg["experiment"]["wandb"] = _use_wandb  # Disable wandb in evaluation mode
if _use_wandb:
    cfg["experiment"]["wandb_kwargs"] = {
        "project": "3DInspection_NoRNN",  # Name of the project in WandB dashboard
        "name": experiment_name,           # Name of this specific run
        "tags": ["PPO", "IsaacLab", args_cli.task]
    }
agent = PPO(models=models, 
            memory=memory,
            cfg=cfg,
            observation_space = env.observation_space,
            action_space=env.action_space,
            device=env.device,)
# path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1862000.pt"
# agent.load(path)
cfg_trainer ={"timesteps": total_timesteps,  # total timesteps to train the agent
                "headless": _headless,
               }#  "stochastic_evaluation": False

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
print("[INFO] Starting training...")
print_dict(cfg_trainer, nesting=4)
if is_eval: 
    print("[INFO] Running in evaluation mode. No training will be performed.")
    path = "/home/tosin/logs/skrl/3DInspection_direct/2026-01-18_18-56-03_ppo_gru_128/checkpoints/agent_155000.pt"
    agent.load(path)
    trainer.eval()
else:
    # path = "/home/tosin/IsaacLab_inspection/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2025-09-07_11-42-53_ppo_gru_128/checkpoints/agent_450000.pt"
    # agent.load(path)
    trainer.train()

