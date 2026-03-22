import os

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

# data class for configurations
class visualisation_mode:
    OCCUPANCY = "occupancy"
    VISIBILITY = "visibility"

class debug_Cfg:
    debug = True
    min_episode_length: int = 1700
    logging_interval: int = 1500
    max_episode_length: int = 1700
    inspection_goal =  0.95
    visualisation_mode = visualisation_mode.OCCUPANCY
    display_ray_counts = True
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = True
    use_wandb =  False #not debug
    headless = False
    num_envs = 1
    nav_camera_modality = "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True

class train_Cfg:
    debug = False
    min_episode_length: int = 1000
    max_episode_length: int = 2500
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

class eval_Cfg:
    debug = True
    max_episode_length: int = 800
    min_episode_length: int = 800
    logging_interval: int = 1500
    inspection_goal =  1.2
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 1
    nav_camera_modality = "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    randomize_spawns = True
    use_hardest_curriculum = True
    rl_camera_width = 84
    rl_camera_height = 84

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

class record_depth_Cfg:
    debug = False
    min_episode_length: int = 2500
    logging_interval: int = 100
    max_episode_length: int = 2500
    inspection_goal =  0.95
    visualisation_mode = None
    display_ray_counts = False
    visualise_point_cloud = False
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = True
    use_wandb =  False 
    headless = False
    num_envs = 1
    data_recording_path = os.path.join(ISAACLAB_ROOT, "data/recorded_depth_data_eval")
    save_images = False
    save_depth = True
    save_interval = 5 # Save every 5 steps to avoid huge data
    nav_camera_modality = "rgbd" 
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    add_high_res_inspection_camera = False
    high_res_camera_width = 512
    high_res_camera_height = 512

modes = [debug_Cfg, #0
    train_Cfg, #1
    eval_Cfg, #2
    record_Cfg, #3
    record_depth_Cfg] #4
cfg_mode = modes[1]