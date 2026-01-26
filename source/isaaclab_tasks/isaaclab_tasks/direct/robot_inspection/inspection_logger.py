import torch
import numpy as np
from collections import deque, defaultdict
import wandb

class InspectionLogger:
    def __init__(self, cfg, use_wandb: bool = False, debug: bool = False):
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.debug = debug
        
        # Buffers
        self.episode_log_buffer = {
            "coverage_percent": deque(maxlen=self.cfg.scene.num_envs * 4),
            "faces_discovered": deque(maxlen=self.cfg.scene.num_envs * 4),
            "mean_inspection_quality": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_map_entropy": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_unique_visible_cell_count": deque(maxlen=self.cfg.scene.num_envs * 4),
            "final_visited_cells_count": deque(maxlen=self.cfg.scene.num_envs * 4),
            "curriculum/current_threshold": deque(maxlen=self.cfg.scene.num_envs * 4),
            "visible_faces_per_step": deque(maxlen=self.cfg.scene.num_envs * 100), # Track visible faces per step
        }
        self.reward_logging_buffer = defaultdict(list)
        
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
                "episode_summary/max_faces_discovered_so_far": self.global_max_faces_discovered,
                "episode_summary/avg_faces_visible_per_step": mean_visible_faces,
            }
            
            for reward_name, reward_values in self.reward_logging_buffer.items():
                if len(reward_values) > 0:
                    log_data[reward_name] = np.mean(reward_values)
            
            # Clear the reward logging buffer after logging
            self.reward_logging_buffer.clear()
            
            if self.use_wandb:
                wandb.log(log_data)
                
    def cache_rewards(self, reward_dict: dict):
        for k, v in reward_dict.items():
            self.reward_logging_buffer[k].append(v)
            
    def update_episode_stats(self, faces_discovered_count: int):
        if faces_discovered_count > self.global_max_faces_discovered:
            self.global_max_faces_discovered = faces_discovered_count

    def log_visible_faces(self, count: float):
        self.episode_log_buffer["visible_faces_per_step"].append(count)
