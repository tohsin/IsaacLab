# data class for configurations
class visualisation_mode:
    OCCUPANCY = "occupancy"
    VISIBILITY = "visibility"

class debug_Cfg:
    debug = True
    min_episode_length: int = 1200
    logging_interval: int = 1500
    max_episode_length: int = 1500
    initaltion_pool_sz : int = 1
    initaltion_pool_sz_goal : int = 1
    inspection_goal =  0.8
    visualisation_mode = None
    display_ray_counts = True
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = False
    visualise_all_objective_spawns = True # Visualise all spawn positions for debug
    visualise_objective_spawns_count :int = 9
    use_wandb =  False #not debug
    headless = False
    num_envs = 1
    nav_camera_modality = "rgb"

class train_Cfg:
    debug = False
    min_episode_length: int = 800
    max_episode_length: int = 1500
    logging_interval: int = 1000
    initaltion_pool_sz : int = 6
    initaltion_pool_sz_goal : int =6
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


class eval_Cfg:
    debug = True
    max_episode_length: int = 1500
    logging_interval: int = 1500
    initaltion_pool_sz : int = 7
    inspection_goal =  0.99
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 1
    nav_camera_modality = "rgb"

class record_Cfg:
    debug = False
    min_episode_length: int = 900
    logging_interval: int = 100
    max_episode_length: int = 900
    initaltion_pool_sz : int = 1
    initaltion_pool_sz_goal : int = 1
    inspection_goal =  1.2
    visualisation_mode = None
    display_ray_counts = False
    visualise_point_cloud = False
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = False
    visualise_all_objective_spawns = False
    visualise_objective_spawns_count :int = 1
    use_wandb =  False 
    headless = False
    num_envs = 1
    data_recording_path = "data/recorded_trajectory"
    save_images = True
    save_depth = False
    save_interval = 2
    nav_camera_modality = "rgb" # "rgb" or "depth"

class record_point_cloud_Cfg:
    debug = False
    min_episode_length: int = 1500
    logging_interval: int = 100
    max_episode_length: int = 1500
    initaltion_pool_sz : int = 1
    initaltion_pool_sz_goal : int = 1
    inspection_goal =  1.2
    visualisation_mode = None
    display_ray_counts = False
    visualise_point_cloud = False
    visualise_face_ids = False
    display_cameras = False
    enable_voxel_visualization = False
    visualise_all_objective_spawns = False
    visualise_objective_spawns_count :int = 1
    use_wandb =  False 
    headless = False
    num_envs = 1
    data_recording_path = "data/recorded_point_clouds"
    save_images = False
    save_depth = True
    save_interval = 5 # Save every 5 steps to avoid huge data
    nav_camera_modality = "rgbd" 

modes = [debug_Cfg, train_Cfg, eval_Cfg, record_Cfg, record_point_cloud_Cfg]
cfg_mode =   modes[1]