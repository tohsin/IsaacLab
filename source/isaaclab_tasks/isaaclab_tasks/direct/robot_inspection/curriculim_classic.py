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
    def __init__(self,
                 init_inspection_threshold = 0.5,
                 max_inspection_threshold = 0.9,
                 curriculum_difficulty_increment = 0.05,
                 default_spatial_milestone: float = 0.8,
                 final_spatial_milestone: float = 0.9,
                 init_spatial_level = 0,
                num_steps: int = 100,
                num_envs: int = 2,
                device: str = None):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.current_level = 0
        self.init_inspection_threshold = init_inspection_threshold
        self.max_inspection_threshold = max_inspection_threshold
        self.curriculum_difficulty_increment = curriculum_difficulty_increment
        self.device = device
        

        self.init_z = 0.06
        self.start_pos = [
                    # [-10.0, 27.6, self.init_z, DEG_0], #Valid # at the back
                    [-1.0, 5.0, self.init_z, DEG_90], #Valid
                    [4.7, 7.4, self.init_z, DEG_NEG_90], # Valid
                    [1.7818, 8.0840, 0.0635,  DEG_UNIQ], #Valid
                    [3.67, 3.87, self.init_z, DEG_NEG_180], #Valid

                    [1.46, 4.18, self.init_z, DEG_NEG_205], #Valid
                    [4.56, 5.2789, self.init_z, DEG_NEG_105], #Valid

                    [-1.0, 7.0, self.init_z, DEG_75], #valid
                    [1.7818, 8.0840, 0.0635,  DEG_UNIQ], #Valid

                    
                    [0, 0, self.init_z, DEG_90],

                    [2.2, 9.4, self.init_z, DEG_0],
                    [2.2, 12.7, self.init_z, DEG_NEG_90],

                    [-1.19, 0.18, self.init_z, DEG_90],
                    [-2.41, 7.47, self.init_z, DEG_0],
                    
                    [-6.93, 4.59, self.init_z, DEG_0],
                    [-10.41, 7.47, self.init_z, DEG_NEG_90],

                    [-20.466, 4.53, self.init_z, DEG_90],
                    [-22, 7.78, self.init_z, DEG_NEG_90],

                    [-24.2, 11.41, self.init_z, DEG_NEG_90]]
        self.episode_length_schedule = [
            1200, 1200,  # Levels 0, 1
            1200, 1200,

            1200, 1200,
            1200, 1200,

            1200, 1200,
            1200, 1200,

            1200, 2200, # Levels 12, 13
            2200, 2200, # Levels 2, 3

            2500, 2500, # Levels 4, 5
            2500, 2500, # Levels 6, 7

            2500, 2500, # Levels 8, 9
            2500, 2500 , # Levels 10, 11 (full length)

            2500, 2500,
            2500, 2500,

            2500, 2500,
            2500, 2500,
            2500


        ]
        self.spatial_level = init_spatial_level
        
        positions = torch.tensor([[item[0], item[1], item[2]] for item in self.start_pos], device=device)
        orientations = torch.tensor([item[3] for item in self.start_pos], device=device)
        self.start_positions_tensor = positions
        self.start_orientations_tensor = orientations
        self.default_spatial_milestone = default_spatial_milestone
        self.final_spatial_milestone = final_spatial_milestone
        self.initialise_task_curriculum()
    #Task curriculum
    def initialise_task_curriculum(self):
        self.inspection_curriculum_level = self.init_inspection_threshold
        self.success_buffer = deque(maxlen=self.num_steps * self.num_envs)
        self.curriculum_threshold = 0.75  # steps
        self.min_episodes_for_curriculum = (self.num_steps - 10) * self.num_envs
        self.success_rate = 0.0

    def get_inspection_level(self):
        return self.inspection_curriculum_level

    def get_current_episode_length(self):
        """Returns the max episode length for the current spatial level."""
        # Ensure we don't go out of bounds if spatial_level exceeds schedule length
        level_index = min(self.spatial_level, len(self.episode_length_schedule) - 1)
        return self.episode_length_schedule[level_index]

    def update_inspection_level(self, episode_successes: torch.Tensor):
        for success in episode_successes:
            self.success_buffer.append(1 if success.item() else 0)

        if len(self.success_buffer) < self.min_episodes_for_curriculum:
            return 

        self.success_rate = sum(self.success_buffer) / len(self.success_buffer)

        if self.spatial_level >= len(self.start_pos) - 4:
            current_milestone = self.final_spatial_milestone
        else:
            current_milestone = self.default_spatial_milestone

        #check if we need to advance spatial level
        if self.inspection_curriculum_level>=current_milestone and self.spatial_level < len(self.start_pos) - 1:
            self.spatial_level += 1
            self.success_buffer.clear()
            self.inspection_curriculum_level = max(self.init_inspection_threshold, self.inspection_curriculum_level - 0.1)
            return

        if self.success_rate >= self.curriculum_threshold and self.inspection_curriculum_level < self.max_inspection_threshold:
            new_threshold = self.inspection_curriculum_level + self.curriculum_difficulty_increment
            self.inspection_curriculum_level = min(new_threshold, self.max_inspection_threshold)
            self.success_buffer.clear()

    def get_start_pos(self, num_resets: int):
        pool_size = self.spatial_level + 1
        random_indices = torch.randint(0, pool_size, (num_resets,), device=self.device)
        new_pos = self.start_positions_tensor[random_indices].to(self.device)
        new_quat = self.start_orientations_tensor[random_indices].to(self.device)
        return new_pos, new_quat
