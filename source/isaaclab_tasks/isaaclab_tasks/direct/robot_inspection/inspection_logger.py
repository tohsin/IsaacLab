import torch
import numpy as np
from collections import deque, defaultdict
import wandb

class InspectionLogger:
    def __init__(self, cfg, use_wandb: bool = False, debug: bool = False, window_size: int = None):
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.debug = debug
        
        # Determine buffer size
        # Default to num_envs * 4 if not provided (old behavior)
        if window_size is None:
             window_size = self.cfg.scene.num_envs * 4
        
        # Buffers
        self.episode_log_buffer = {
            "coverage_percent": deque(maxlen=window_size),
            "faces_discovered": deque(maxlen=window_size),
            "mean_inspection_quality": deque(maxlen=window_size),
            "final_map_entropy": deque(maxlen=window_size),
            "final_unique_visible_cell_count": deque(maxlen=window_size),
            "final_visited_cells_count": deque(maxlen=window_size),
            "curriculum/current_threshold": deque(maxlen=window_size),
            "visible_faces_per_step": deque(maxlen=window_size), # Track visible faces per step (maybe keep this one dynamic/shorter? No, let's align.)
            "curriculum/active_obstacles": deque(maxlen=window_size),
            "episode_summary/final_map_entropy_percent": deque(maxlen=window_size),
            "curriculum/task_area": deque(maxlen=window_size),
        }
        self.reward_logging_buffer = defaultdict(list)
        
        # New buffers for per-episode cumulative sums
        # We'll initialize these dynamically on the first update to adapt to device
        self.episode_cumulative_rewards = defaultdict(lambda: torch.zeros(self.cfg.scene.num_envs, device='cuda:0')) 
        
        # Derived metrics state
        self.global_max_faces_discovered = 0
        
    def log_step(self, common_step_counter: int, curriculum_success_rate: float):
        if not self.debug and common_step_counter % self.cfg.logging_interval == 0:
            mean_coverage = np.mean(self.episode_log_buffer["coverage_percent"]) if self.episode_log_buffer["coverage_percent"] else 0.0
            mean_faces_discovered = np.mean(self.episode_log_buffer["faces_discovered"]) if self.episode_log_buffer["faces_discovered"] else 0.0
            mean_inspection_quality = np.mean(self.episode_log_buffer["mean_inspection_quality"]) if self.episode_log_buffer["mean_inspection_quality"] else 0.0
            mean_final_map_entropy = np.mean(self.episode_log_buffer["final_map_entropy"]) if self.episode_log_buffer["final_map_entropy"] else 0.0
            mean_final_visited_cells_count = np.mean(self.episode_log_buffer["final_visited_cells_count"]) if self.episode_log_buffer["final_visited_cells_count"] else 0.0
            mean_final_unique_visible_cell_count = np.mean(self.episode_log_buffer["final_unique_visible_cell_count"]) if self.episode_log_buffer["final_unique_visible_cell_count"] else 0.0
            mean_current_threshold = np.mean(self.episode_log_buffer["curriculum/current_threshold"]) if self.episode_log_buffer["curriculum/current_threshold"] else 0.0            
            mean_visible_faces = np.mean(self.episode_log_buffer["visible_faces_per_step"]) if self.episode_log_buffer["visible_faces_per_step"] else 0.0
            
            mean_final_map_entropy_percent = np.mean(self.episode_log_buffer["episode_summary/final_map_entropy_percent"]) if self.episode_log_buffer["episode_summary/final_map_entropy_percent"] else 0.0
            mean_task_area = np.mean(self.episode_log_buffer["curriculum/task_area"]) if self.episode_log_buffer["curriculum/task_area"] else 0.0
            
            # Calculate means for reward sums
            reward_sum_metrics = {}
            for k, buffer in self.episode_log_buffer.items():
                if k.startswith("reward_sum/"):
                     reward_sum_metrics[k] = np.mean(buffer) if buffer else 0.0

            log_data = {
                # Curriculum Status
                "curriculum/exploration_threshold_goal": mean_current_threshold,
                "curriculum/success_rate": curriculum_success_rate,                # Episode Performance Summary
                "episode_summary/mean_coverage_percent": mean_coverage,
                "episode_summary/mean_faces_discovered": mean_faces_discovered,
                "episode_summary/mean_inspection_quality": mean_inspection_quality,
                "episode_summary/mean_final_map_entropy": mean_final_map_entropy,
                "episode_summary/mean_final_unique_visible_cell_count": mean_final_unique_visible_cell_count,
                "episode_summary/mean_final_visited_cells_count": mean_final_visited_cells_count,
                "episode_summary/mean_final_map_entropy_percent": mean_final_map_entropy_percent,
                "episode_summary/max_faces_discovered_so_far": self.global_max_faces_discovered,
                "episode_summary/avg_faces_visible_per_step": mean_visible_faces,
                "curriculum/active_obstacles": np.mean(self.episode_log_buffer["curriculum/active_obstacles"]) if self.episode_log_buffer.get("curriculum/active_obstacles") else 0.0,
                "curriculum/task_area": mean_task_area,
            }
            
            # Add reward sum metrics
            log_data.update(reward_sum_metrics)
            
            for reward_name, reward_values in self.reward_logging_buffer.items():
                if len(reward_values) > 0:
                    log_data[reward_name] = np.mean(reward_values)
            
            # Clear the reward logging buffer after logging
            self.reward_logging_buffer.clear()
            
            if self.use_wandb:
                wandb.log(log_data)
    
    def accumulate_rewards(self, reward_dict: dict):
        """
        Accumulates rewards per environment for the current step (for episode sums)
        AND logs the mean step reward (for instantaneous rates).
        reward_dict: Dictionary of {name: tensor(num_envs)}
        """
        for k, v in reward_dict.items():
            # 1. Accumulate Sums (Per-Episode)
            if k not in self.episode_cumulative_rewards:
                 self.episode_cumulative_rewards[k] = torch.zeros_like(v)
            self.episode_cumulative_rewards[k] += v

            # 2. Log Step Mean (Instantaneous)
            # We prefix with "reward_components/" to keep it organized
            # v is a tensor of shape (num_envs,), we take the mean across envs
            self.reward_logging_buffer[f"reward_components/{k}"].append(v.mean().item())

    def log_and_reset_episode_rewards(self, env_ids):
        """
        Logs the cumulative rewards for the reset environments and resets their counters.
        env_ids: Indices of environments being reset.
        """
        if len(env_ids) == 0:
            return

        for k, v in self.episode_cumulative_rewards.items():
            # Extract sums for the reset environments
            sums = v[env_ids]
            
            # Log these sums into the episode log buffer (which feeds into log_step means)
            # We want to log the "Mean Sum per Episode" effectively.
            # Convert to list and extend our deque
            
            # Key modification: separate scaled vs raw if needed, but here k captures that distinction
            # We add a prefix "reward_sum/" for clarity if not present, though caller likely provides good keys.
            
            # Ensure the deque exists in episode_log_buffer
            buffer_key = f"reward_sum/{k}"
            if buffer_key not in self.episode_log_buffer:
                 self.episode_log_buffer[buffer_key] = deque(maxlen=self.cfg.scene.num_envs * 4)

            for val in sums.cpu().tolist():
                self.episode_log_buffer[buffer_key].append(val)
            
            # Reset
            v[env_ids] = 0.0
                
    def cache_rewards(self, reward_dict: dict):
        for k, v in reward_dict.items():
            self.reward_logging_buffer[k].append(v)
            
    def update_episode_stats(self, faces_discovered_count: int):
        if faces_discovered_count > self.global_max_faces_discovered:
            self.global_max_faces_discovered = faces_discovered_count

    def log_visible_faces(self, count: float):
        self.episode_log_buffer["visible_faces_per_step"].append(count)

    def clear_episode_buffers(self):
        """
        Clears all episode-level log buffers and cumulative rewards.
        This is useful when the curriculum level changes, to avoid smoothing 
        metrics across different difficulty levels.
        """
        for buffer in self.episode_log_buffer.values():
            buffer.clear()
        self.episode_cumulative_rewards.clear()
