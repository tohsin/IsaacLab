import argparse
import os
import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import ipdb
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

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model, GaussianMixin
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
class SharedFeatureExtractor( Model):
    def __init__(self,
                observation_space,
                action_space,
                sequence_length=64, 
                num_envs=1, 
                device=None):
        Model.__init__(self, observation_space, action_space)
        camera_space = observation_space.spaces["cameras"]
        robot_pose_space = observation_space.spaces["robot-pose"]

        self.num_envs = num_envs


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
        self.feature_size = camera_cnn_output_dim + self.robot_pose_dim

    def unflatten_observations(self, flat_obs):
        """
        Manually unflatten the observation tensor back to camera and robot pose components.
        """
        batch_size = flat_obs.shape[0]
        cam_end = self.camera_flat_size
        
        camera_flat = flat_obs[:, :cam_end]
        robot_pose = flat_obs[:, cam_end:]
        
        camera_reshaped = camera_flat.view(batch_size, *self.camera_shape)
        camera_obs = camera_reshaped.permute(0, 3, 1, 2)

        return camera_obs, robot_pose

    def compute(self, inputs, role):
        states = inputs["states"]
        camera_obs, robot_pose = self.unflatten_observations(states)
        image_features = self.camera_features_extractor(camera_obs)
        combined_features = torch.cat((image_features, robot_pose), dim=1)  # (batch, cnn_output_dim + robot_pose_dim)
        return combined_features
    
class Policy(GaussianMixin, Model):
    def __init__(
                self,
                observation_space, 
                action_space, device, 
                num_envs,
                feature_extractor: SharedFeatureExtractor,
                clip_actions=True,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.feature_extractor = feature_extractor
        self.device = device
        self.num_envs = num_envs

        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions)
        )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        features = self.feature_extractor.compute(inputs, role)
        mean_actions = self.policy_head(features)
        # log_std = torch.clamp(self.log_std_parameter, self.min_log_std, self.max_log_std)
        # return_value = (mean_actions, log_std, {"rnn": [hidden_states]})
        return mean_actions, self.log_std_parameter, {}


class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 feature_extractor : SharedFeatureExtractor, num_envs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.feature_extractor = feature_extractor
        self.device = device
        self.num_envs = num_envs
        
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size + self.num_actions, 512), # Takes features AND actions
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    
    def compute(self, inputs, role):
        features = self.feature_extractor.compute(inputs, role)
        # Concatenate features with actions
        taken_actions = inputs["taken_actions"].to(self.device)
        combined = torch.cat([features, taken_actions], dim=1)

        value_estimate = self.value_head(combined), {}
        return value_estimate


env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env_cfg.seed = 42
env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)

device = env.device
replay_buffer_size = 3_000
# assume num env is 16
memory_size = replay_buffer_size // env.num_envs
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

shared_features_extractor = SharedFeatureExtractor(observation_space=env.observation_space, 
                                                   action_space=env.action_space, device=device)
models = {}
models["policy"] = Policy(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, num_envs=env.num_envs, feature_extractor=shared_features_extractor)
models["critic_1"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, num_envs=env.num_envs, feature_extractor=shared_features_extractor)
models["critic_2"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, num_envs=env.num_envs, feature_extractor=shared_features_extractor)
models["target_critic_1"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, num_envs=env.num_envs, feature_extractor=shared_features_extractor)
models["target_critic_2"] = QNetwork(
                            observation_space=env.observation_space, action_space=env.action_space,
                            device=device, num_envs=env.num_envs, feature_extractor=shared_features_extractor)

cfg =  SAC_DEFAULT_CONFIG.copy()
cfg["batch_size"] = 128
cfg["gradient_steps"] = 1

cfg["polyak"] = 0.005
cfg["discount_factor"] = 0.99

cfg["actor_learning_rate"] = 1e-5
cfg["critic_learning_rate"] = 5e-5


cfg["learning_rate_scheduler"] = None
cfg["learning_rate_scheduler_kwargs"] = {}


cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["random_timesteps"] =  3_000
cfg["learning_starts"] =  3_000


cfg["grad_norm_clip"] = 1.0

cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-5
cfg["initial_entropy_value"] = 1.0
cfg["target_entropy"] = None

cfg["rewards_shaper"] = None
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
cfg["experiment"]["wandb"] = True  # Enable wandb logging

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

