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
                max_coverage_ratio: float = 0.99,
                coverage_increment: float = 0.05,

                start_quality_threshold: float = 0.1,
                max_quality_threshold: float = 0.6,
                quality_increment: float = 0.02,
                
                num_envs: int = 2,
                device: str = None):
        
        # 
        self.current_coverage_threshold = start_coverage_ratio
        self.max_coverage_threshold = max_coverage_ratio
        self.coverage_increment = coverage_increment
        self.initaltion_pool_sz_goal = cfg_mode.initaltion_pool_sz_goal

        self.current_quality_treshold = start_quality_threshold
        self.max_quality_threshold = max_quality_threshold
        self.quality_increment = quality_increment
        
        self.min_episode_length_limit = cfg_mode.min_episode_length
        self.max_episode_length_limit = cfg_mode.max_episode_length

        self.num_envs = num_envs 
        self.device = device
        self.success_buffer = deque(maxlen=50 * self.num_envs) # Buffer ~20 resets per env
        self.quality_buffer = deque(maxlen=50 * self.num_envs)
        self.min_episodes_for_update = 15 * self.num_envs

        self.success_rate_threshold = 0.60 
        self.success_rate = 0.0 
        
        self.last_objective_pool_size = -1
        self.last_robot_pool_size = -1

        self._setup_spawn_points()
        print("--- Inspection Curriculum Initialized (No Time Schedule) ---")
        print(f"  Initial Coverage Goal: {self.current_coverage_threshold*100:.1f}%")
        print("----------------------------------------------------------")

    def _setup_spawn_points(self):
        self.init_z = 0.06
        self.init_z_goal = 0.4
        self.start_pos_robot = [ # X and Y positions only
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
                    # [-7.0, -8.81], # Bottom-Left (Stress Test)
                    ]

        self.start_pos_objective = [ # X and Y positions only
                    [0.0, -2.0], 
                    [0.0, 0], #Valid # at the back
                    [0, -5], 
                    [-4, -5], 
                    [1.15, -5.56], 
                    [-5.5, 1.74], 
                    [-7.0, -8.8],
                    [5, -7.4 ],
                    [3.63, 6.32],
                    [-1.08, 6.32],
                    [5.7, 13.67],
                    [0.83, 15.10],
                    [-4.7, 10.2],
                    # [5.7, 13.67],   # Top-Right (Stress Test)
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
        self.start_positions_tensor  = torch.tensor(
            [[item[0], item[1], self.init_z] for item in self.start_pos_robot],
            device=self.device
        )
        self.objective_positions_tensor = torch.tensor(
            [[item[0], item[1], self.init_z_goal] for item in self.start_pos_objective],
            device=self.device
        )

    #Task curriculum
    def get_current_coverage_goal(self) -> float:
        """Returns the current required % of faces (0.0 to 1.0)"""
        return self.current_coverage_threshold
    def get_current_quality_goal(self) -> float:
        """Returns the current required mean quality (0.0 to 1.0)"""
        return self.current_quality_treshold

    def update_curriculum(self, episode_successes: torch.Tensor, episode_qualities: torch.Tensor):
        """
        Updates the curriculum based on success.
        """
        # Add results to buffer
        for success, quality in zip(episode_successes, episode_qualities):
            self.success_buffer.append(1 if success.item() else 0)
            self.quality_buffer.append(quality.item())

        # Wait until buffer is full enough
        if len(self.success_buffer) < self.min_episodes_for_update:
            return

        self.success_rate = sum(self.success_buffer) / len(self.success_buffer)
        self.avg_quality = sum(self.quality_buffer) / len(self.quality_buffer)

        # Check if we should advance
        # Check if we should advance
        if self.success_rate >= self.success_rate_threshold:
            # Increase Coverage Requirement
            new_cov = self.current_coverage_threshold + self.coverage_increment
            
            if self.current_coverage_threshold < self.max_coverage_threshold:
                self.current_coverage_threshold = min(new_cov, self.max_coverage_threshold)

            if self.current_coverage_threshold >= new_cov:
                self.success_buffer.clear() # Reset buffer to prove capability at new level
                self.quality_buffer.clear()
                print(f"--- CURRICULUM LEVEL UP ---")
                print(f"  New Coverage Goal: {self.current_coverage_threshold:.2f}")
                print(f"  Prev Success Rate: {self.success_rate:.2f}, Prev Avg Quality: {self.avg_quality:.2f}")
                print(f"  Prev Success Rate: {self.success_rate:.2f}, Prev Avg Quality: {self.avg_quality:.2f}")

    def get_current_episode_length(self) -> int:
        """Returns the current max episode length based on curriculum progress."""
        # Calculate progress ratio (0.0 to 1.0)
        progress = (self.current_coverage_threshold - 0.1) / (self.max_coverage_threshold - 0.1)
        progress = max(0.0, min(1.0, progress))
        
        episode_length = int(self.min_episode_length_limit + (self.max_episode_length_limit - self.min_episode_length_limit) * progress)
        return episode_length

    def get_start_pos(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets random start positions from the pool based on progress."""
        total_items = len(self.start_positions_tensor)
        min_items = cfg_mode.initaltion_pool_sz
        
        # Calculate progress ratio (0.0 to 1.0)
        progress = (self.current_coverage_threshold - 0.1) / (self.max_coverage_threshold - 0.1)
        progress = max(0.0, min(1.0, progress))
        
        # Current active pool size
        pool_size = int(min_items + (total_items - min_items) * progress)
        pool_size = max(min_items, min(total_items, pool_size))
        
        if pool_size != self.last_robot_pool_size:
             print(f"[Curriculum] Robot Spawn Pool Size: {pool_size}/{total_items} (Progress: {progress:.2f})")
             self.last_robot_pool_size = pool_size

        random_pos_indices = torch.randint(0, pool_size, (num_resets,), device=self.device)
        selected_pos = self.start_positions_tensor[random_pos_indices]

        self.allowed_orientations = self.allowed_orientations[:1] if cfg_mode.debug else self.allowed_orientations

        # Random orientations
        num_orientations = len(self.allowed_orientations)
        random_ori_indices = torch.randint(0, num_orientations, (num_resets,), device=self.device)
        selected_ori = self.allowed_orientations[random_ori_indices]

        return selected_pos, selected_ori

    def get_objective_start_pos(self, num_resets: int, robot_pos: torch.Tensor) -> torch.Tensor:
        """
        Gets random start positions for the inspection object.
        Ensures the object is not placed too close to the robot.
        Pool of textends as coverage threshold increases.
        """
        # Determine pool size based on coverage threshold or success rate
        # Scale pool size linearly from min items to all items based on progress
        total_items = len(self.objective_positions_tensor)
        min_items =  self.initaltion_pool_sz_goal
        
        # Calculate progress ratio (0.0 to 1.0)
        progress = (self.current_coverage_threshold - 0.1) / (self.max_coverage_threshold - 0.1)
        progress = max(0.0, min(1.0, progress))
        
        # Current active pool size
        pool_size = int(min_items + (total_items - min_items) * progress)
        pool_size = max(min_items, min(total_items, pool_size))
        
        if pool_size != self.last_objective_pool_size:
            print(f"[Curriculum] Objective Spawn Pool Size: {pool_size}/{total_items} (Progress: {progress:.2f})")
            self.last_objective_pool_size = pool_size

        # Slice the tensor to get available positions
        available_positions = self.objective_positions_tensor[:pool_size]
        
        # Sample positions
        # We need to ensure we don't pick a position too close to the robot
        min_dist_sq = 1.5**2 # Minimum 1.5m distance
        
        selected_positions = torch.zeros((num_resets, 3), device=self.device)
        
        # Naive rejection sampling per environment
        # Since we are doing this for a batch, we can iterate or try to vectorize
        # Given num_resets is usually small (equal to num_envs or subset), loop is acceptable for spawn logic logic
        
        for i in range(num_resets):
            valid = False
            attempts = 0
            while not valid and attempts < 20:
                idx = torch.randint(0, pool_size, (1,), device=self.device)
                candidate = available_positions[idx].squeeze(0)
                
                # Check distance to robot
                # robot_pos[i] is (3,)
                dist_sq = torch.sum((candidate[:2] - robot_pos[i, :2])**2)
                
                if dist_sq > min_dist_sq:
                    selected_positions[i] = candidate
                    valid = True
                attempts += 1
            
            if not valid:
                # Fallback: just take the candidate if we couldn't find a better one
                # Or pick the furthest one? For now just take it to avoid hang
                selected_positions[i] = candidate
                
        return selected_positions
