import argparse
import os
import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
torch.autograd.set_detect_anomaly(True)

from isaaclab.app import AppLauncher
is_eval = False
# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")

# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-Direct-v0", help="Name of the task.")
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
_headless = True
args_cli = parser.parse_args()
args_cli.enable_cameras =  True
args_cli.headless = _headless
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.agents.torch.sac import SAC_RNN as SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model, GaussianMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
# sys.argv.append("--headless")
sys.argv.append("--enable_cameras")

set_seed(42)
# for some reason changing the clip actionsvarialbe to true in thr training script causes this error

    
class Policy(GaussianMixin, Model):
    def __init__(self, 
                 observation_space,
                action_space, 
                device, 
                sequence_length,
                num_envs,
                clip_actions=True,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
      
        self.num_envs = num_envs
        self.gru_hidden_size=128
        self.gru_num_layers=1
        self.device = device
        self.sequence_length = sequence_length
        self.num_envs = num_envs

        camera_space = observation_space.spaces["cameras"]
        robot_pose_space = observation_space.spaces["robot-pose"]

        self.camera_shape = camera_space.shape
        self.robot_pose_dim = robot_pose_space.shape[0]

        self.camera_flat_size = np.prod(self.camera_shape).item()
        self.total_obs_size = self.camera_flat_size + self.robot_pose_dim

        self.camera_features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.camera_shape[-1], out_channels=64, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            #permute(STATES, (0, 3, 1, 2)) 
            dummy_camera_input = torch.zeros(1, self.camera_shape[-1], *self.camera_shape[:2])
            camera_cnn_output_dim = self.camera_features_extractor(dummy_camera_input).shape[1]

        #self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim + map_cnn_output_dim
        self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim 
        # print(f"DEBUG: gru_input_size: {self.gru_input_size}")

        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)
        #output heads

        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions)
        )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
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

        if flat_obs.shape[1] != self.total_obs_size:
            print(f"WARNING: Observation size mismatch! Expected {self.total_obs_size}, got {flat_obs.shape[1]}")
        
        # Split camera and robot pose data
        cam_start, cam_end = 0, self.camera_flat_size
        pose_start = cam_end

        camera_flat = flat_obs[:, cam_start:cam_end]
        robot_pose = flat_obs[:, pose_start:]
        # Reshape camera data from flat to (batch, height, width, channels)
        camera_reshaped = camera_flat.view(batch_size, *self.camera_shape)
        camera_obs = camera_reshaped.permute(0, 3, 1, 2)

        # print(f"DEBUG: camera_obs final shape: {camera_obs.shape}")

        return camera_obs, robot_pose
    
    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]



        camera_obs, robot_pose = self.unflatten_observations(states)
        image_features = self.camera_features_extractor(camera_obs)

        combined_features = torch.cat((image_features, robot_pose), dim=1)  # (batch, cnn_output_dim + robot_pose_dim)

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
        mean_actions = self.policy_head(rnn_output)
        return mean_actions, self.log_std_parameter, {"rnn": [hidden_states]}


class QNetwork(DeterministicMixin, Model):
    def __init__(
                    self,
                    observation_space,
                    action_space,
                    device, 
                    sequence_length,
                    num_envs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.device = device
        self.gru_hidden_size=128
        self.gru_num_layers=1
        self.sequence_length = sequence_length
        self.num_envs = num_envs

        camera_space = observation_space.spaces["cameras"]
        robot_pose_space = observation_space.spaces["robot-pose"]

        self.camera_shape = camera_space.shape
        self.robot_pose_dim = robot_pose_space.shape[0]

        self.camera_flat_size = np.prod(self.camera_shape).item()
        self.total_obs_size = self.camera_flat_size + self.robot_pose_dim

        self.camera_features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.camera_shape[-1], out_channels=64, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            #permute(STATES, (0, 3, 1, 2)) 
            dummy_camera_input = torch.zeros(1, self.camera_shape[-1], *self.camera_shape[:2])
            camera_cnn_output_dim = self.camera_features_extractor(dummy_camera_input).shape[1]

        #self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim + map_cnn_output_dim
        self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim 
        # print(f"DEBUG: gru_input_size: {self.gru_input_size}")

        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True)  # b
        
        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size + self.num_actions, 256), # Takes features AND actions
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def get_specification(self):
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

        if flat_obs.shape[1] != self.total_obs_size:
            print(f"WARNING: Observation size mismatch! Expected {self.total_obs_size}, got {flat_obs.shape[1]}")
        
        # Split camera and robot pose data
        cam_start, cam_end = 0, self.camera_flat_size
        pose_start = cam_end

        camera_flat = flat_obs[:, cam_start:cam_end]
        robot_pose = flat_obs[:, pose_start:]
        # Reshape camera data from flat to (batch, height, width, channels)
        camera_reshaped = camera_flat.view(batch_size, *self.camera_shape)
        camera_obs = camera_reshaped.permute(0, 3, 1, 2)

        return camera_obs, robot_pose
    
    def compute(self, inputs, role):
        states = inputs["states"]
        actions = inputs["taken_actions"] 
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        camera_obs, robot_pose = self.unflatten_observations(states)
        image_features = self.camera_features_extractor(camera_obs)

        combined_features = torch.cat((image_features, robot_pose), dim=1)  # (batch, cnn_output_dim + robot_pose_dim)

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
        value_input = torch.cat((rnn_output, actions), dim=1)
        value_estimate = self.value_head(value_input)
        return value_estimate, {"rnn": [hidden_states]}


env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env_cfg.seed = 42
env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)

device = env.device
replay_buffer_size = 1024
# assume num env is 16
sequence_length = 32
memory_size = replay_buffer_size // env.num_envs
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)


models = {}
models["policy"] = Policy(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, sequence_length=sequence_length, num_envs=env.num_envs,)
models["critic_1"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, sequence_length=sequence_length, num_envs=env.num_envs,)
models["critic_2"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, sequence_length=sequence_length, num_envs=env.num_envs)
models["target_critic_1"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, sequence_length=sequence_length, num_envs=env.num_envs)
models["target_critic_2"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, sequence_length=sequence_length, num_envs=env.num_envs)

cfg =  SAC_DEFAULT_CONFIG.copy()
cfg["batch_size"] = 64
cfg["gradient_steps"] = 1

cfg["polyak"] = 0.005
cfg["discount_factor"] = 0.99

cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4


cfg["learning_rate_scheduler"] = None
cfg["learning_rate_scheduler_kwargs"] = {}


cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["random_timesteps"] =  64
cfg["learning_starts"] =  64


cfg["grad_norm_clip"] = 1.0

cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 0.01
cfg["initial_entropy_value"] = 0.5
cfg["target_entropy"] = None

cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg["mixed_precision"] = False


log_root_path = os.path.join("logs", "skrl", "3DInspection_direct")
log_root_path = os.path.abspath(log_root_path)

experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_gru_128"

print(f"[INFO] Logging experiment in directory: {log_root_path}")

log_dir = os.path.join(log_root_path, experiment_name)
print(f"[INFO] skrl will log this experiment in: {log_dir}")

os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)



cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["name"] = "IsaacLab-scripts_reinforcement_learning_skrl"
cfg["experiment"]["checkpoint_interval"] = 10_000
cfg["experiment"]["directory"] = log_root_path
cfg["experiment"]["experiment_name"] = experiment_name
cfg["experiment"]["wandb"] = True  # Enable wandb

agent =  SAC(models=models, 
            memory=memory,
            cfg=cfg,
            observation_space = env.observation_space,
            action_space=env.action_space,
            device=env.device,)
# path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1862000.pt"
# agent.load(path)
cfg_trainer ={"timesteps": 500_000,  # total timesteps to train the agent
                "headless": _headless,
               }#  "stochastic_evaluation": False

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
print("[INFO] Starting training...")
print_dict(cfg_trainer, nesting=4)
if is_eval: 
    print("[INFO] Running in evaluation mode. No training will be performed.")
    # path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1860000.pt"
    # agent.load(path)
    trainer.eval()
else:
    # path = "/home/tosin/IsaacLab_inspection/scripts/reinforcement_learning/skrl/logs/skrl/3DInspection_direct/2025-09-07_11-42-53_ppo_gru_128/checkpoints/agent_450000.pt"
    # agent.load(path)
    trainer.train()

