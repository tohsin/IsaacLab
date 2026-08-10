import os
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
                coverage_increment_down: float = 0.025,
                success_rate_increase_thresh = 0.7,
                success_rate_decrease_thresh = 0.57,

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

        self.success_buffer = deque(maxlen=2000) # Buffer ~20 resets per env
        self.quality_buffer = deque(maxlen=2000)
        self.min_episodes_for_update = 1600 # 2560

        # Hysteresis Thresholds
        self.success_rate_increase_thresh = success_rate_increase_thresh
        self.success_rate_decrease_thresh = success_rate_decrease_thresh
        
        self.success_rate = 0.0 
        self.avg_quality = 0.0 # Init
        
        self.last_robot_pool_size = -1

        self.is_main_process = int(os.environ.get("REAL_LOCAL_RANK", 0)) == 0

        self._setup_spawn_points()
        if self.is_main_process:
            print("--- Inspection Curriculum Initialized (Reversible w/ Hysteresis) ---")
            print(f"  Initial Coverage Goal: {self.current_coverage_threshold*100:.1f}%")
            print(f"  Increase Thresh: >{self.success_rate_increase_thresh:.2f}, Decrease Thresh: <{self.success_rate_decrease_thresh:.2f}")
            print("----------------------------------------------------------")

    def _setup_spawn_points(self):
        self.init_z = 0.06
        self.init_z_goal = 0.3
        
        # Fixed room coordinates for continuous randomized spawning
        self.spawn_min_x = -5.0
        self.spawn_max_x = -self.spawn_min_x 
        self.spawn_min_y = -5.0
        self.spawn_max_y_init = -self.spawn_min_y
        self.spawn_max_y_final = 8.0


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

        # This method is driven by local environment resets, which are not synchronized
        # across distributed workers. Keep the curriculum statistics local so one rank
        # cannot enter a collective while another rank is reducing model gradients.
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
                     if self.is_main_process:
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
                     if self.is_main_process:
                         print(f"--- CURRICULUM LEVEL DOWN ---")
                         print(f"  New Coverage Goal: {self.current_coverage_threshold:.2f}")
                         print(f"  Reason: Success Rate ({self.success_rate:.2f}) < {self.success_rate_decrease_thresh}")

    def get_progress(self) -> float:
        """Returns the curriculum progress from 0.0 to 1.0."""
        if getattr(cfg_mode, "use_hardest_curriculum", False):
            return 1.0
        if self.max_coverage_threshold == self.start_coverage_ratio:
            return 0.0
        progress = (self.current_coverage_threshold - self.start_coverage_ratio) / (self.max_coverage_threshold - self.start_coverage_ratio)
        return max(0.0, min(1.0, progress))

    def get_current_episode_length(self) -> int:
        """Returns the current max episode length based on curriculum progress."""
        progress = self.get_progress()
        
        episode_length = int(self.min_episode_length_limit + (self.max_episode_length_limit - self.min_episode_length_limit) * progress)
        return episode_length

    def get_current_max_crashes(self) -> int:
        """Returns the current max allowed crashes based on curriculum progress."""
        progress = self.get_progress()
        start_crashes = getattr(cfg_mode, "start_crashes", 5)
        end_crashes = getattr(cfg_mode, "end_crashes", 1)
        current_crashes = int(round(start_crashes - progress * (start_crashes - end_crashes)))
        return max(end_crashes, current_crashes)

    def get_current_spawn_max_y(self) -> float:
        """Returns the current maximum Y spawn coordinate based on curriculum progress."""
        progress = self.get_progress()
        current_y = self.spawn_max_y_init + (self.spawn_max_y_final - self.spawn_max_y_init) * progress
        min_y = getattr(cfg_mode, "min_spawn_max_y", self.spawn_max_y_init)
        return max(min_y, current_y)

    def get_total_task_area(self) -> float:
        """Returns the total square meters of the current valid spawn area bounds."""
        max_y = self.get_current_spawn_max_y()
        width_x = self.spawn_max_x - self.spawn_min_x
        height_y = max_y - self.spawn_min_y
        return float(width_x * height_y)

    def get_num_active_obstacles(self, max_obstacles: int) -> int:
        """Returns the number of active obstacles based on progress (0.0 to 1.0)"""
        if getattr(cfg_mode, "is_simplified", False):
            return 0
            
        # If we are using fixed spawns or hardest curriculum, show all obstacles
        if getattr(cfg_mode, "use_hardest_curriculum", False) or getattr(cfg_mode, "fixed_spawns", False):
            return max_obstacles
            
        progress = self.get_progress()
        # Scale up faster, starting with a base of 1 obstacle at 0.0 progress, which makes it 2 at 0.1, up to max_obstacles
        num = int(progress * max_obstacles) + 1
        
        min_obs = getattr(cfg_mode, "min_obstacles", 2)
        return max(min_obs, min(num, max_obstacles))

    def get_start_pos(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets random start positions from bounding boxes."""
        # --- Hardcoded Spawn for Data Recording or Debugging ---
        if (getattr(cfg_mode, "data_recording_path", None) is not None or getattr(cfg_mode, "fixed_spawns", False)) and not getattr(cfg_mode, "randomize_spawns", False):
            # Robot at [0, 0, z]
            pos = torch.zeros((num_resets, 3), device=self.device)
            pos[:, 0] = 0.0
            pos[:, 1] = 0.0
            pos[:, 2] = self.init_z
            
            # Identity quaternion (w=1, x=0, y=0, z=0)
            ori = torch.zeros((num_resets, 4), device=self.device)
            ori[:, 0] = 1.0 
            return pos, ori
            
        current_spawn_max_y = self.get_current_spawn_max_y()
        pos_x = self.spawn_min_x + torch.rand((num_resets,), device=self.device) * (self.spawn_max_x - self.spawn_min_x)
        pos_y = self.spawn_min_y + torch.rand((num_resets,), device=self.device) * (current_spawn_max_y - self.spawn_min_y)
        
        selected_pos = torch.zeros((num_resets, 3), device=self.device)
        selected_pos[:, 0] = pos_x
        selected_pos[:, 1] = pos_y
        selected_pos[:, 2] = self.init_z

        # Random orientations (uniform yaw around Z-axis)
        yaw = torch.rand((num_resets,), device=self.device) * 2 * torch.pi
        if getattr(cfg_mode, "debug", False):
            yaw = torch.zeros_like(yaw)  # Face forward in debug mode
            
        selected_ori = torch.zeros((num_resets, 4), device=self.device)
        selected_ori[:, 0] = torch.cos(yaw / 2)  # w
        selected_ori[:, 3] = torch.sin(yaw / 2)  # z

        return selected_pos, selected_ori

    def get_objective_start_pos(self, num_resets: int, robot_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets random start positions for the inspection object within bounds.
        Ensures the object is not placed too close to the robot.
        """
        # --- Hardcoded Spawn for Data Recording or Debugging ---
        if (getattr(cfg_mode, "data_recording_path", None) is not None or getattr(cfg_mode, "fixed_spawns", False)) and not getattr(cfg_mode, "randomize_spawns", False):
            # Objective at [0, -2.0, z_goal]
            pos = torch.zeros((num_resets, 3), device=self.device)
            pos[:, 0] = 0.0
            pos[:, 1] = -2.0
            pos[:, 2] = self.init_z_goal
            
            ori = torch.zeros((num_resets, 4), device=self.device)
            ori[:, 0] = 1.0 
            return pos, ori
        
        min_dist_to_obj = getattr(cfg_mode, "min_dist_to_objective", 1.5)
        min_dist_sq = min_dist_to_obj**2
        selected_positions = torch.zeros((num_resets, 3), device=self.device)
        selected_positions[:, 2] = self.init_z_goal
        
        current_spawn_max_y = self.get_current_spawn_max_y()
        # Naive rejection sampling per environment
        for i in range(num_resets):
            valid = False
            attempts = 0
            curr_robot_pos = robot_pos[i, :2]
            
            while not valid and attempts < 100:
                pos_x = self.spawn_min_x + torch.rand(1, device=self.device).item() * (self.spawn_max_x - self.spawn_min_x)
                pos_y = self.spawn_min_y + torch.rand(1, device=self.device).item() * (current_spawn_max_y - self.spawn_min_y)
                
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
                selected_positions[i, 1] = current_spawn_max_y

        # For the object, we use identity quaternion to not break the raycaster
        selected_ori = torch.zeros((num_resets, 4), device=self.device)
        selected_ori[:, 0] = 1.0  # w
                
        return selected_positions, selected_ori

    def get_obstacle_start_pos(self, num_resets: int, existing_positions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets random start positions for an obstacle within bounds using batched sampling.
        Ensures the obstacle is not placed too close to existing positions.
        """
        if (getattr(cfg_mode, "data_recording_path", None) is not None or getattr(cfg_mode, "fixed_spawns", False)) and not getattr(cfg_mode, "randomize_spawns", False):
            pos = torch.zeros((num_resets, 3), device=self.device)
            # When we have many obstacles (like 10 in our dynamic dataset), spawning them all around (2,2) with a tiny jitter overlaps them heavily or hides them inside each other.
            # Instead, let's distribute them in a grid or circle reliably.
            idx = len(existing_positions) - 2 # Offset by 2 since existing_positions has [robot_pos, objective_pos] already
            
            # Grid layout distributing evenly across the highest area
            num_cols = 4
            num_rows = 3
            row = idx // num_cols
            col = idx % num_cols
            
            margin_x = 1.5
            margin_y = 1.5
            span_x = (self.spawn_max_x - margin_x) - (self.spawn_min_x + margin_x)
            span_y = (self.spawn_max_y_final - margin_y) - (self.spawn_min_y + margin_y)
            
            pos[:, 0] = self.spawn_min_x + margin_x + col * (span_x / max(1, num_cols - 1))
            pos[:, 1] = self.spawn_min_y + margin_y + row * (span_y / max(1, num_rows - 1))
            pos[:, 2] = self.init_z_goal
            
            ori = torch.zeros((num_resets, 4), device=self.device)
            ori[:, 0] = 1.0 
            return pos, ori

        min_dist = getattr(cfg_mode, "min_dist_between_obstacles", 2.0)
        selected_positions = torch.zeros((num_resets, 3), device=self.device)
        selected_positions[:, 2] = self.init_z_goal
        current_spawn_max_y = self.get_current_spawn_max_y()
        
        # Track which environments still need a valid position
        needs_spawn = torch.ones(num_resets, dtype=torch.bool, device=self.device)
        
        max_attempts = 100
        for _ in range(max_attempts):
            if not needs_spawn.any():
                break
                
            num_needed = needs_spawn.sum().item()
            
            # Batch generate proposals
            proposal_x = self.spawn_min_x + torch.rand(num_needed, device=self.device) * (self.spawn_max_x - self.spawn_min_x)
            proposal_y = self.spawn_min_y + torch.rand(num_needed, device=self.device) * (current_spawn_max_y - self.spawn_min_y)
            proposals = torch.stack([proposal_x, proposal_y], dim=-1)
            
            # Check collisions with all existing entities
            valid = torch.ones(num_needed, dtype=torch.bool, device=self.device)
            for existing_pos in existing_positions:
                # Get the relevant existing pos for the envs that need spawn
                relevant_existing = existing_pos[needs_spawn, :2]
                dists = torch.norm(proposals - relevant_existing, dim=-1)
                valid &= (dists > min_dist)
            
            # For the ones that are valid, assign them
            valid_indices_in_needed = valid.nonzero().squeeze(-1)
            # Map back to original env indices
            needed_indices = needs_spawn.nonzero().squeeze(-1)
            actual_valid_indices = needed_indices[valid_indices_in_needed]
            
            selected_positions[actual_valid_indices, 0] = proposals[valid_indices_in_needed, 0]
            selected_positions[actual_valid_indices, 1] = proposals[valid_indices_in_needed, 1]
            needs_spawn[actual_valid_indices] = False
            
        # Fallback for any that didn't find a spot
        if needs_spawn.any():
            fallback_indices = needs_spawn.nonzero().squeeze(-1)
            print(f"[Curriculum WARN] Could not find valid spawn for {len(fallback_indices)} obstacles. Using fallback.")
            selected_positions[fallback_indices, 0] = self.spawn_min_x
            selected_positions[fallback_indices, 1] = current_spawn_max_y
            
        # For obstacles, we use identity quaternion (w=1) since they are symmetric or fine in default pose
        selected_ori = torch.zeros((num_resets, 4), device=self.device)
        selected_ori[:, 0] = 1.0  # w
        
        return selected_positions, selected_ori
