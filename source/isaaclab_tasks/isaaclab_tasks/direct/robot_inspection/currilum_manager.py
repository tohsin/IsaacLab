#View logs
DEG_0 = [1.0, 0.0, 0.0, 0.0]
DEG_UNIQ =[9.9791e-01, 5.4488e-07, 2.1990e-06, 6.4633e-02]
DEG_90 = [0.7071068, 0.0, 0.0, 0.7071068]
DEG_75 = [0.7933533,  0, 0, 0.6087614]
DEG_NEG_90 = [0.7071068, 0.0, 0.0, -0.7071068]
DEG_NEG_180 = [0.0, 0.0, 0.0, -1.0]
DEG_NEG_205 = [ -0.2164396 , 0, 0, -0.976296]
DEG_NEG_105 = [ 0.6087614,  0, 0, -0.7933533]
import torch
from collections import deque

class Curriculum:
    def __init__(
                self,
                total_map_cells = 60_000,
                assumed_surface_cell_ratio: float = 0.1,
                start_exploration_ratio: float = 0.2,
                max_exploration_ratio: float = 0.80,
                exploration_increment_ratio: float = 0.05,
                num_envs: int = 2,
                device: str = None):
        

        total_surface_cells = total_map_cells * assumed_surface_cell_ratio
        self.init_exploration_threshold = total_surface_cells * start_exploration_ratio
        self.max_exploration_threshold = total_surface_cells * max_exploration_ratio
        self.exploration_increment = total_surface_cells * exploration_increment_ratio

        self.exploration_threshold = self.init_exploration_threshold
        self.temporal_level = 0

        self.num_envs = num_envs
        self.current_level = 0
        
        self.device = device
        self.success_buffer = deque(maxlen=20 * self.num_envs) # Buffer ~20 resets per env
        self.min_episodes_for_update = 10 * self.num_envs
        self.success_rate_threshold_for_increment = 0.70  # If >70% success, increase count threshold
        self.success_rate_threshold_for_temporal = 0.85 # If >85% success, increase episode time
        self.success_rate_threshold = 0.70
        self.success_rate = 0.0
       
        self.episode_length_schedule = [1100, 1400, 1800, 2000, 2400]
        milestone_ratios = [0.20, 0.35, 0.50, 0.65, 0.81]
        self.exploration_milestones = [total_surface_cells * r for r in milestone_ratios]

        self._setup_spawn_points()
        print("--- Simplified Exploration Curriculum Initialized ---")
        print(f"  Target Surface Cells: {total_surface_cells:.0f}")
        print(f"  Initial Cell Count Threshold: {self.init_exploration_threshold:.0f}")
        print(f"  Maximum Cell Count Threshold: {self.max_exploration_threshold:.0f}")
        print(f"  Cell Count Increment: {self.exploration_increment:.0f}")
        print(f"  Exploration Milestones: {[int(m) for m in self.exploration_milestones]}")
        print("--------------------------------------------------")

    def _setup_spawn_points(self):
        self.init_z = 0.06
        self.start_pos = [
                    [0, 0, self.init_z, DEG_0], #Valid # at the back
                    [0, -5, self.init_z, DEG_NEG_180], 
                    [-4.47, -5, self.init_z, DEG_90], 
                    [1.15, -5.56, self.init_z, DEG_90], #Valid
                    [1.15, 0, self.init_z, DEG_90], # Valid
                    [-7.0, 1.74, self.init_z, DEG_0], # Navigation required Starts here

                    [-7.0, -8.81,self.init_z, DEG_90 ],
                    [6.7, -8.81, self.init_z, DEG_NEG_90 ],
                    [3.63, 6.32, self.init_z, DEG_NEG_90 ],
                    [-1.08, 6.32, self.init_z, DEG_NEG_90 ],
                    [-6.73, 15.10, self.init_z, DEG_0 ],
                    [0.83, 15.10, self.init_z, DEG_90 ],
                    ]
        self.start_pos = self.start_pos[:1]
        positions = torch.tensor([[item[0], item[1], item[2]] for item in self.start_pos], device=self.device)
        orientations = torch.tensor([item[3] for item in self.start_pos], device=self.device)
        self.start_positions_tensor = positions
        self.start_orientations_tensor = orientations

    #Task curriculum
    def get_exploration_threshold(self) -> float:
            return self.exploration_threshold

    def get_current_episode_length(self) -> int:
        """Returns the max episode length for the current temporal level."""
        level_index = min(self.temporal_level, len(self.episode_length_schedule) - 1)
        return self.episode_length_schedule[level_index]

    def update_exploration_level(self, episode_successes: torch.Tensor):
        """Updates the curriculum based on the success of completed episodes."""
        for success in episode_successes:
            self.success_buffer.append(1 if success.item() else 0)

        if len(self.success_buffer) < self.min_episodes_for_update:
            return

        self.success_rate = sum(self.success_buffer) / len(self.success_buffer)

        # --- Curriculum Advancement Logic ---
        if self.success_rate >= self.success_rate_threshold:
                # 1. Increase the exploration goal
                new_threshold = self.exploration_threshold + self.exploration_increment
                self.exploration_threshold = min(new_threshold, self.max_exploration_threshold)
                self.success_buffer.clear() # Reset success buffer after an update
                print(f"--- SUCCESS: INCREASING EXPLORATION THRESHOLD TO {self.exploration_threshold:.0f} ---")

                # 2. Check if the new goal crosses a milestone, and increase time if it does
                if self.temporal_level < len(self.episode_length_schedule) - 1:
                    next_milestone = self.exploration_milestones[self.temporal_level + 1]
                    if self.exploration_threshold >= next_milestone:
                        self.temporal_level += 1
                        print(f"--- MILESTONE REACHED! ADVANCING TEMPORAL LEVEL TO {self.temporal_level} ---")
                        print(f"    New Episode Length: {self.get_current_episode_length()}")

    def get_start_pos(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets random start positions from the ENTIRE pool of spawn points."""
        pool_size = len(self.start_positions_tensor)
        random_indices = torch.randint(0, pool_size, (num_resets,), device=self.device)
        new_pos = self.start_positions_tensor[random_indices]
        new_quat = self.start_orientations_tensor[random_indices]
        return new_pos, new_quat
