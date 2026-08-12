import os

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

# data class for configurations
class map_view_mode:
    GLOBAL = "global"
    LOCAL = "local"


class map_channels:
    OCCUPANCY = "occupancy"
    VISIBILITY = "visibility"
    VISITATION = "visitation"
    COLLISION = "collision"

class visualisation_mode:
    def __init__(self,
                channel=map_channels.OCCUPANCY, 
                map_mode=map_view_mode.LOCAL):
        self.channel = channel
        self.map_mode = map_mode

class debug_Cfg:
    debug = True
    min_episode_length: int = 500
    logging_interval: int = 1500
    max_episode_length: int = 1500
    inspection_goal =  0.95
    visualisation_mode = visualisation_mode(channel=map_channels.COLLISION, map_mode=map_view_mode.LOCAL)
    display_ray_counts = True
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = True
    visualize_env_id = 0
    use_wandb =  False #not debug
    headless = False
    num_envs = 8
    nav_camera_modality = "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = False
    reset_on_crash = False
    collision_consecutive_steps: int = 2

class train_Cfg_base:
    debug = False
    min_episode_length: int = 1000
    max_episode_length: int = 2300
    logging_interval: int = 1000
    inspection_goal =  0.1
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  True #not debug
    headless = True
    num_envs = 128
    nav_camera_modality = "rgbd" # "rgb", "depth", or "rgbd"
    use_depth_mask = False
    use_optical_flow_as_quality = True
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = False
    reset_on_crash = True
    start_crashes: int = 5
    end_crashes: int = 1
    # Debounce contact noise. This is not a collision budget: a confirmed
    # collision terminates immediately after this many consecutive detections.
    collision_consecutive_steps: int = 2
    min_obstacles: int = 2
    max_obstacles: int = 6
    min_spawn_max_y: float = 5.0
    min_dist_to_objective: float = 2.0
    min_dist_between_obstacles: float = 2.2

class train_Cfg_pretrain(train_Cfg_base):
    min_episode_length: int = 500
    max_episode_length: int = 1250 # New for ~10Hz
    # max_episode_length: int = 2500 # Old for ~21.5Hz
    

class train_Cfg_finetune(train_Cfg_base):
    inspection_goal =  0.90
    min_episode_length: int = 2300
    # max_episode_length: int = 2300
    # min_obstacles: int = 
    # min_spawn_max_y: float = 9.0

class eval_Cfg:
    debug = False
    max_episode_length: int = 1250
    min_episode_length: int = 1250
    logging_interval: int = 1500
    inspection_goal =  0.9
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 128
    nav_camera_modality = "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    randomize_spawns = True
    use_hardest_curriculum = True
    reset_on_crash = True
    enable_voxel_visualization = False

class record_Cfg:
    debug = False
    min_episode_length: int = 900
    logging_interval: int = 100
    max_episode_length: int = 900
    inspection_goal =  1.2
    visualisation_mode = None
    display_ray_counts = False
    visualise_point_cloud = False
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = False
    use_wandb =  False 
    headless = False
    num_envs = 32
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_trajectory")
    save_images = True
    save_depth = False
    save_interval = 2
    nav_camera_modality = "rgb" # "rgb" or "depth"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    randomize_spawns = True
    use_hardest_curriculum = True
    reset_on_crash = False

class record_depth_Cfg:
    debug = False
    min_episode_length: int = 1250
    logging_interval: int = 100
    max_episode_length: int = 1250
    inspection_goal =  1.0
    visualisation_mode = None
    display_ray_counts = False
    visualise_point_cloud = False
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = True
    use_wandb =  False 
    headless = True
    num_envs = 1
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval/rubiks_cube")
    save_images = False
    save_depth = True
    save_interval = 5 # Save every 5 steps to avoid huge data
    nav_camera_modality = "rgbd" 
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    reset_on_crash = False
    add_high_res_inspection_camera = True
    high_res_camera_width = 512
    high_res_camera_height = 512

modes = [debug_Cfg, #0
    train_Cfg_pretrain, #1
    train_Cfg_finetune, #2
    eval_Cfg, #3
    record_Cfg, #4
    record_depth_Cfg] #5
cfg_mode = modes[1]
