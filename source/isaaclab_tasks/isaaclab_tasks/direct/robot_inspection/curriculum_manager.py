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
from .run_config import cfg_mode
class Curriculum:
    def __init__(
                self,
                start_coverage_ratio: float = cfg_mode.inspection_goal,
                max_coverage_ratio: float = 0.95,
                coverage_increment: float = 0.05,
                num_envs: int = 2,
                device: str = None):
        
        # 
        self.current_coverage_threshold = start_coverage_ratio
        self.max_coverage_threshold = max_coverage_ratio
        self.coverage_increment = coverage_increment


        self.num_envs = num_envs 
        self.device = device
        self.success_buffer = deque(maxlen=200 * self.num_envs) # Buffer ~20 resets per env
        self.min_episodes_for_update = 10 * self.num_envs

        self.success_rate_threshold = 0.60 
        self.success_rate = 0.0 
        
        self._setup_spawn_points()
        print("--- Inspection Curriculum Initialized (No Time Schedule) ---")
        print(f"  Initial Coverage Goal: {self.current_coverage_threshold*100:.1f}%")
        print("----------------------------------------------------------")

    def _setup_spawn_points(self):
        self.init_z = 0.06
        self.start_pos = [ # X and Y positions only
                    [0.0, 0], #Valid # at the back
                    [0, -5], 
                    [-4.47, -5], 
                    [1.15, -5.56], 
                    [1.15, 0],
                    [-7.0, 1.74], # Navigation required Starts here
                    [-7.0, -8.81],
                    [6.7, -8.81 ],
                    [3.63, 6.32],
                    [-1.08, 6.32],
                    [-6.73, 15.10],
                    [0.83, 15.10],
                    ]
        self.allowed_orientations = torch.tensor([
            DEG_0, 
            DEG_UNIQ, 
            DEG_90, 
            DEG_75, 
            DEG_NEG_90, 
            DEG_NEG_180, 
            DEG_NEG_205, 
            DEG_NEG_105
        ], device=self.device, dtype=torch.float32)
        self.start_pos = self.start_pos[:cfg_mode.initaltion_pool_sz]
        self.start_positions_tensor  = torch.tensor(
            [[item[0], item[1], self.init_z] for item in self.start_pos],
            device=self.device
        )

    #Task curriculum
    def get_current_coverage_goal(self) -> float:
        """Returns the current required % of faces (0.0 to 1.0)"""
        return self.current_coverage_threshold

    def update_curriculum(self, episode_successes: torch.Tensor):
        """
        Updates the curriculum based on success.
        """
        # Add results to buffer
        for success in episode_successes:
            self.success_buffer.append(1 if success.item() else 0)

        # Wait until buffer is full enough
        if len(self.success_buffer) < self.min_episodes_for_update:
            return

        self.success_rate = sum(self.success_buffer) / len(self.success_buffer)

        # Check if we should advance
        if self.success_rate >= self.success_rate_threshold:
            # Increase Coverage Requirement
            new_threshold = self.current_coverage_threshold + self.coverage_increment
            
            if self.current_coverage_threshold < self.max_coverage_threshold:
                self.current_coverage_threshold = min(new_threshold, self.max_coverage_threshold)
                self.success_buffer.clear() # Reset buffer to prove capability at new level
                print(f"--- CURRICULUM LEVEL UP: New Coverage Goal {self.current_coverage_threshold:.2f} ---")

    def get_start_pos(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets random start positions from the ENTIRE pool of spawn points."""
        pool_size = len(self.start_positions_tensor)
        random_pos_indices = torch.randint(0, pool_size, (num_resets,), device=self.device)
        selected_pos = self.start_positions_tensor[random_pos_indices]
        self.allowed_orientations = self.allowed_orientations[:1] if cfg_mode.debug else self.allowed_orientations

        # Random orientations
        num_orientations = len(self.allowed_orientations)
        random_ori_indices = torch.randint(0, num_orientations, (num_resets,), device=self.device)
        selected_ori = self.allowed_orientations[random_ori_indices]

        return selected_pos, selected_ori
