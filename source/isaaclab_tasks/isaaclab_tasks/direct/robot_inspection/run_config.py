# data class for configurations
class visualisation_mode:
    OCCUPANCY = "occupancy"
    VISIBILITY = "visibility"

class debug_Cfg:
    debug = True
    max_episode_length: int = 1200
    initaltion_pool_sz : int = 1
    inspection_goal =  0.8
    visualisation_mode = visualisation_mode.OCCUPANCY
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    display_cameras = False
    use_wandb =  False #not debug
    headless = False
    num_envs = 1

class train_Cfg:
    debug = False
    max_episode_length: int = 1200
    initaltion_pool_sz : int = 7
    inspection_goal =  0.05
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    display_cameras = False
    use_wandb =  True #not debug
    headless = True
    num_envs = 128


class eval_Cfg:
    debug = True
    max_episode_length: int = 1200
    initaltion_pool_sz : int = 7
    inspection_goal =  0.95
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 1


modes = [debug_Cfg, train_Cfg, eval_Cfg]
cfg_mode =   modes[2]