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
                
                # Asymmetric increments
                coverage_increment_up: float = 0.05,
                coverage_increment_down: float = 0.02,
                success_rate_increase_thresh = 0.67,
                success_rate_decrease_thresh = 0.58,

                start_quality_threshold: float = 0.03,
                max_quality_threshold: float = 0.6,
                quality_increment: float = 0.02,
                
                num_envs: int = 2,
                device: str = None):
        
        # 
        self.current_coverage_threshold = start_coverage_ratio
        self.start_coverage_ratio = start_coverage_ratio # Keep track of min
        self.max_coverage_threshold = max_coverage_ratio
        
        self.coverage_increment_up = coverage_increment_up
        self.coverage_increment_down = coverage_increment_down
        
        self.current_quality_treshold = start_quality_threshold
        self.max_quality_threshold = max_quality_threshold
        self.quality_increment = quality_increment
        
        self.min_episode_length_limit = cfg_mode.min_episode_length
        self.max_episode_length_limit = cfg_mode.max_episode_length

        self.num_envs = num_envs 
        self.device = device
        # self.success_buffer = deque(maxlen= 70 * self.num_envs) # Buffer ~20 resets per env
        # self.quality_buffer = deque(maxlen= 70 * self.num_envs) #2240
        # self.min_episodes_for_update = 50 * self.num_envs # 
        
        # self.success_buffer = deque(maxlen=50 * self.num_envs) # Buffer ~20 resets per env
        # self.quality_buffer = deque(maxlen=50 * self.num_envs) #6400
        # self.min_episodes_for_update = 20 * self.num_envs # 2560

        self.success_buffer = deque(maxlen=2500) # Buffer ~20 resets per env
        self.quality_buffer = deque(maxlen=2500)
        self.min_episodes_for_update = 1400 # 2560

        # Hysteresis Thresholds
        self.success_rate_increase_thresh = success_rate_increase_thresh
        self.success_rate_decrease_thresh = success_rate_decrease_thresh
        
        self.success_rate = 0.0 
        self.avg_quality = 0.0 # Init
        
        self.last_objective_pool_size = -1
        self.last_robot_pool_size = -1

        self._setup_spawn_points()
        print("--- Inspection Curriculum Initialized (Reversible w/ Hysteresis) ---")
        print(f"  Initial Coverage Goal: {self.current_coverage_threshold*100:.1f}%")
        print(f"  Increase Thresh: >{self.success_rate_increase_thresh:.2f}, Decrease Thresh: <{self.success_rate_decrease_thresh:.2f}")
        print("----------------------------------------------------------")

    def _setup_spawn_points(self):
        self.init_z = 0.06
        self.init_z_goal = 0.3
        
        # Fixed room coordinates for continuous randomized spawning
        self.spawn_min_x = -4.0
        self.spawn_max_x = -self.spawn_min_x 
        self.spawn_min_y = -4.0
        self.spawn_max_y = -self.spawn_min_y

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

        # Check if we should advance (Increase Difficulty)
        if self.success_rate >= self.success_rate_increase_thresh:
            new_cov = self.current_coverage_threshold + self.coverage_increment_up
            
            # Use a small epsilon for float comparison or just ensure we don't go over max
            if self.current_coverage_threshold < self.max_coverage_threshold:
                 # Only clear buffer if we actually change something
                 if new_cov > self.current_coverage_threshold:
                     self.current_coverage_threshold = min(new_cov, self.max_coverage_threshold)
                     self.success_buffer.clear()
                     self.quality_buffer.clear()
                     print(f"--- CURRICULUM LEVEL UP ---")
                     print(f"  New Coverage Goal: {self.current_coverage_threshold:.2f}")
                     print(f"  Reason: Success Rate ({self.success_rate:.2f}) >= {self.success_rate_increase_thresh}")

        # Check if we should retreat (Decrease Difficulty)
        elif self.success_rate < self.success_rate_decrease_thresh:
             new_cov = self.current_coverage_threshold - self.coverage_increment_down
            
             if self.current_coverage_threshold > self.start_coverage_ratio:
                 # Only clear buffer if we actually change something
                 if new_cov < self.current_coverage_threshold:
                     self.current_coverage_threshold = max(new_cov, self.start_coverage_ratio)
                     self.success_buffer.clear()
                     self.quality_buffer.clear()
                     print(f"--- CURRICULUM LEVEL DOWN ---")
                     print(f"  New Coverage Goal: {self.current_coverage_threshold:.2f}")
                     print(f"  Reason: Success Rate ({self.success_rate:.2f}) < {self.success_rate_decrease_thresh}")

    def get_progress(self) -> float:
        """Returns the curriculum progress from 0.0 to 1.0."""
        if self.max_coverage_threshold == self.start_coverage_ratio:
            return 0.0
        progress = (self.current_coverage_threshold - self.start_coverage_ratio) / (self.max_coverage_threshold - self.start_coverage_ratio)
        return max(0.0, min(1.0, progress))

    def get_current_episode_length(self) -> int:
        """Returns the current max episode length based on curriculum progress."""
        progress = self.get_progress()
        
        episode_length = int(self.min_episode_length_limit + (self.max_episode_length_limit - self.min_episode_length_limit) * progress)
        return episode_length

    def get_start_pos(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets random start positions from bounding boxes."""
        # --- Hardcoded Spawn for Data Recording or Debugging ---
        if getattr(cfg_mode, "data_recording_path", None) is not None or getattr(cfg_mode, "fixed_spawns", False):
            # Robot at [0, 0, z]
            pos = torch.zeros((num_resets, 3), device=self.device)
            pos[:, 0] = 0.0
            pos[:, 1] = 0.0
            pos[:, 2] = self.init_z
            
            # Identity quaternion (w=1, x=0, y=0, z=0)
            ori = torch.zeros((num_resets, 4), device=self.device)
            ori[:, 0] = 1.0 
            return pos, ori
            
        pos_x = self.spawn_min_x + torch.rand((num_resets,), device=self.device) * (self.spawn_max_x - self.spawn_min_x)
        pos_y = self.spawn_min_y + torch.rand((num_resets,), device=self.device) * (self.spawn_max_y - self.spawn_min_y)
        
        selected_pos = torch.zeros((num_resets, 3), device=self.device)
        selected_pos[:, 0] = pos_x
        selected_pos[:, 1] = pos_y
        selected_pos[:, 2] = self.init_z

        self.allowed_orientations = self.allowed_orientations[:1] if getattr(cfg_mode, "debug", False) else self.allowed_orientations

        # Random orientations
        num_orientations = len(self.allowed_orientations)
        random_ori_indices = torch.randint(0, num_orientations, (num_resets,), device=self.device)
        selected_ori = self.allowed_orientations[random_ori_indices]

        return selected_pos, selected_ori

    def get_objective_start_pos(self, num_resets: int, robot_pos: torch.Tensor) -> torch.Tensor:
        """
        Gets random start positions for the inspection object within bounds.
        Ensures the object is not placed too close to the robot.
        """
        # --- Hardcoded Spawn for Data Recording or Debugging ---
        if getattr(cfg_mode, "data_recording_path", None) is not None or getattr(cfg_mode, "fixed_spawns", False):
            # Objective at [0, -2.0, z_goal]
            pos = torch.zeros((num_resets, 3), device=self.device)
            pos[:, 0] = 0.0
            pos[:, 1] = -2.0
            pos[:, 2] = self.init_z_goal
            return pos
        
        min_dist_sq = 1.5**2 # Minimum 2m distance
        selected_positions = torch.zeros((num_resets, 3), device=self.device)
        selected_positions[:, 2] = self.init_z_goal
        
        # Naive rejection sampling per environment
        for i in range(num_resets):
            valid = False
            attempts = 0
            curr_robot_pos = robot_pos[i, :2]
            
            while not valid and attempts < 100:
                pos_x = self.spawn_min_x + torch.rand(1, device=self.device).item() * (self.spawn_max_x - self.spawn_min_x)
                pos_y = self.spawn_min_y + torch.rand(1, device=self.device).item() * (self.spawn_max_y - self.spawn_min_y)
                
                dist_sq = (pos_x - curr_robot_pos[0].item())**2 + (pos_y - curr_robot_pos[1].item())**2
                
                if dist_sq > min_dist_sq:
                    selected_positions[i, 0] = pos_x
                    selected_positions[i, 1] = pos_y
                    valid = True
                attempts += 1
                
            # Last Resort: If we exhaust 100 attempts, just place it far enough deterministically
            if not valid:
                print(f"[Curriculum WARN] Could not find valid spawn for env {i} far enough from robot. Using fallback.")
                selected_positions[i, 0] = self.spawn_min_x
                selected_positions[i, 1] = self.spawn_max_y
                
        return selected_positions
