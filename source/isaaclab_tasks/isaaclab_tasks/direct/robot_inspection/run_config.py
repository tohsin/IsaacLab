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
    use_wandb =  False #not debug
    headless = False
    num_envs = 1

class train_Cfg:
    debug = False
    min_episode_length: int = 1200
    max_episode_length: int = 1500
    logging_interval: int = 1000
    initaltion_pool_sz : int = 12
    initaltion_pool_sz_goal : int = 13
    inspection_goal =  0.1
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  True #not debug
    headless = True
    num_envs = 128


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


modes = [debug_Cfg, train_Cfg, eval_Cfg]
cfg_mode =   modes[0]