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
    egocentric_map = True
    min_episode_length: int = 500
    logging_interval: int = 1500
    max_episode_length: int = 10_000
    inspection_dataset = "primitive"
    inspection_target = "sphere"
    inspection_goal =  0.95
    visualisation_mode = visualisation_mode(channel=map_channels.OCCUPANCY, map_mode=map_view_mode.LOCAL)
    display_ray_counts = True
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_cameras = True
    enable_voxel_visualization = True
    visualise_task_area = False
    visualize_env_id = 0
    use_wandb =  False #not debug
    headless = False
    nav_camera_modality = "rgbd"
    ptz_camera_modality = "rgbd"
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    max_obstacles: int = 15
    reset_on_crash = True
    collision_consecutive_steps: int = 2
    is_simplified = True

class train_Cfg_base: # For pretriaing as a base
    debug = False
    egocentric_map = True
    min_episode_length: int = 500
    max_episode_length: int = 1250
    logging_interval: int = 1000
    inspection_goal =  0.1
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  True #not debug
    headless = True
    nav_camera_modality = "rgbd" # "rgb", "depth", or "rgbd"
    ptz_camera_modality = "rgbd" # "rgb", "depth", or "rgbd"

    enable_depth_sensor_noise = True
    depth_pixel_dropout_prob = 0.01
    depth_pixel_std_dev_multiplier = 0.01

    enable_rgb_sensor_noise = True
    rgb_noise_std = 0.05
    rgb_color_jitter_brightness = 0.2
    rgb_color_jitter_contrast = 0.2
    rgb_color_jitter_saturation = 0.2
    rgb_color_jitter_hue = 0.1

    enable_semantic_mask_noise = True
    semantic_mask_false_negative_prob = 0.01
    semantic_mask_false_positive_prob = 0.001

    use_depth_mask = False
    min_inspection_distance = 0.7
    use_optical_flow_as_quality = True
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = False
    reset_on_crash = True
    start_crashes: int = 1
    end_crashes: int = 1
    # Debounce contact noise. This is not a collision budget: a confirmed
    # collision terminates immediately after this many consecutive detections.
    collision_consecutive_steps: int = 2
    min_obstacles: int = 3
    max_obstacles: int = 15
    min_spawn_max_y: float = 5.0
    min_dist_to_objective: float = 2.0

    use_radius_aware_obstacle_spawning: bool = False
    # Radius-aware spawn clearances. Required center distance is the sum of
    # both footprint radii and the applicable free-surface clearance.
    robot_footprint_radius: float = 0.45
    fallback_target_footprint_radius: float = 0.8
    target_obstacle_surface_clearance: float = 0.45
    obstacle_obstacle_surface_clearance: float = 0.50
    robot_obstacle_surface_clearance: float = 0.40
    # Backward-compatible center-distance fallback for callers that do not
    # provide footprint radii.
    min_dist_between_obstacles: float = 2.2

class eval_Cfg:
    debug = False
    # Evaluate the UR10 mount under the same target physics, sensor noise,
    # spawning, obstacle, and collision settings used by the primitive test.
    # inspection_dataset = "primitive"
    # inspection_target = "tessellated_t_block"
    inspection_dataset = "evaluation"
    inspection_target = "ur10_mount"
    kinematic_inspection_target = True
    inspection_target_mass = 1000.0

    egocentric_map = True

    enable_depth_sensor_noise = True
    depth_pixel_dropout_prob = 0.01
    depth_pixel_std_dev_multiplier = 0.01

    enable_rgb_sensor_noise = True
    rgb_noise_std = 0.05
    rgb_color_jitter_brightness = 0.2
    rgb_color_jitter_contrast = 0.2
    rgb_color_jitter_saturation = 0.2
    rgb_color_jitter_hue = 0.1

    enable_semantic_mask_noise = True
    semantic_mask_false_negative_prob = 0.01
    semantic_mask_false_positive_prob = 0.001

    max_episode_length: int = 1250
    min_episode_length: int = 1250
    logging_interval: int = 1500
    inspection_goal =  0.95
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 1 # Single environment for easier debugging
    nav_camera_modality = "rgbd"
    ptz_camera_modality = "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = True
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    start_crashes: int = 1
    end_crashes: int = 1
    collision_consecutive_steps: int = 2
    max_obstacles: int = 15
    reset_on_crash = True
    enable_voxel_visualization = False

    use_radius_aware_obstacle_spawning: bool = False
    min_dist_between_obstacles: float = 2.2
    min_dist_to_objective: float = 2.0

    is_simplified = False
    robot_footprint_radius: float = 0.45
    fallback_target_footprint_radius: float = 0.8
    target_obstacle_surface_clearance: float = 0.45
    obstacle_obstacle_surface_clearance: float = 0.50
    robot_obstacle_surface_clearance: float = 0.40



class record_depth_Cfg(eval_Cfg):
    debug = False

    #inspection target information
    kinematic_inspection_target = False
    inspection_target_mass = 100_000.0
    inspection_dataset = eval_Cfg.inspection_dataset
    inspection_target = eval_Cfg.inspection_target

    egocentric_map = True
    min_episode_length: int = 1250
    logging_interval: int = 100
    max_episode_length: int = 1250

    enable_depth_sensor_noise = True
    depth_pixel_dropout_prob = 0.01
    depth_pixel_std_dev_multiplier = 0.01

    enable_semantic_mask_noise = True
    semantic_mask_false_negative_prob = 0.01
    semantic_mask_false_positive_prob = 0.001
    # Point-cloud coverage is evaluated offline. Keep the face-based goal above
    # 100% so it cannot terminate a recording early and bias trajectory length.
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
    data_recording_path = os.path.join(
        ISAACLAB_ROOT,
        "data/recorded_depth_data_eval",
        inspection_target or inspection_dataset,
    )
    # RGB image sequences are only needed for GS/NeRF-style reconstruction.
    save_images = False
    # Keep an episode-level visual trace without retaining individual RGB files.
    save_video = True
    save_depth = True
    record_all_episodes = True
    create_timestamped_run = True
    save_interval = 5 # Save every 5 steps to avoid huge data
    nav_camera_modality = "rgbd" 
    ptz_camera_modality = "rgbd" 
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    # End and label crashed episodes so crash rate and time-to-crash can be
    # reported from the same evaluation run.
    reset_on_crash = True
    add_high_res_inspection_camera = True
    high_res_camera_width = 256
    high_res_camera_height = 256

modes = [debug_Cfg, #0
    train_Cfg_base, #1
    eval_Cfg, #2
    record_depth_Cfg] #3
cfg_mode = modes[1]

    # record_Cfg, #3





class record_Cfg:
    debug = False
    egocentric_map = False
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
    ptz_camera_modality = "rgb" # "rgb", "depth", or "rgbd"
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = False
    randomize_spawns = True
    use_hardest_curriculum = True
    reset_on_crash = False
