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

class policy_map_channel_sets:
    """Fixed-shape policy observation sets for controlled map ablations."""

    OCCUPANCY_ONLY = (map_channels.OCCUPANCY,)
    OCCUPANCY_VISIBILITY = (map_channels.OCCUPANCY, map_channels.VISIBILITY)
    ALL = (
        map_channels.OCCUPANCY, map_channels.VISIBILITY, map_channels.VISITATION
    )

class visualisation_mode:
    def __init__(self,
                channel=map_channels.OCCUPANCY, 
                map_mode=map_view_mode.LOCAL):
        self.channel = channel
        self.map_mode = map_mode

eval_dataset = {
    "extinguisher": "extinguisher", # verified
    "barrel": "barrel", #verified
    "dolly": "dolly", # verified
    "bracket": "small_corner_bracket_physics", #verified
    "ur10": "ur10_mount", #veified
    "pallet": "pallet", # verified
    "caster": "caster", #verified
}
class debug_Cfg:
    debug = True
    enable_pt_actuation: bool = True
    # Optical zoom changes rendered USD camera state at runtime. Keep it off
    # by default; enabling it changes the policy from 4 to 5 actions.
    enable_camera_zoom: bool = False
    enable_map_ray_chunking: bool = False
    map_ray_chunk_percent: float = 12.5
    egocentric_map = True
    min_episode_length: int = 500
    logging_interval: int = 1500
    max_episode_length: int = 10_000
    # inspection_dataset = "primitive"
    # inspection_target = "tessellated_thin_legged_body"
    inspection_dataset = "evaluation"
    inspection_target = eval_dataset["ur10"]
    # Stable, repeatable pose for checking imported scale, orientation, and
    # ground contact before quantitative evaluation.
    kinematic_inspection_target = True
    inspection_target_mass = 1000.0
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
    tipover_max_tilt_degrees: float = 45.0
    tipover_consecutive_steps: int = 2
    is_simplified = False
    use_radius_aware_obstacle_spawning: bool = True
    robot_footprint_radius: float = 0.35
    fallback_target_footprint_radius: float = 0.8
    target_obstacle_surface_clearance: float = 2.0
    obstacle_obstacle_surface_clearance: float = 1.5
    robot_obstacle_surface_clearance: float = 0.8

    # Directional safety shield. The local occupancy map is already expressed
    # in the robot frame, so this does not change the policy observation or the
    # checkpoint architecture.
    use_directional_safety_shield: bool = False
    safety_shield_occupancy_threshold: float = 1.1
    safety_shield_robot_radius: float = 0.35
    safety_shield_margin: float = 0.10
    safety_shield_horizon_s: float = 0.50
    safety_shield_prediction_steps: int = 6
    safety_shield_z_min: float = 0.10
    safety_shield_z_max: float = 0.80
    safety_shield_linear_scales: tuple = (1.0, 0.75, 0.50, 0.25, 0.0)

class train_Cfg_base: # For pretriaing as a base
    debug = False
    # Fixed-camera ablation: remove pan/tilt from the policy action space and
    # hold both joints at their neutral reset pose.
    enable_pt_actuation: bool = False
    # False restores the stable, fixed-35-mm, four-action baseline. Set True
    # only when training/evaluating a checkpoint that includes the zoom action.
    enable_camera_zoom: bool = False
    # Split each filtered mapping point cloud into roughly eight Warp launches.
    # This shortens individual kernels; clamping and synchronization still
    # happen once after all chunks.
    enable_map_ray_chunking: bool = True
    map_ray_chunk_percent: float = 25.0
    egocentric_map = True
    # Keep the complete baseline map observation for the camera ablation.
    policy_map_channels = policy_map_channel_sets.ALL
    min_episode_length: int = 500
    max_episode_length: int = 1200
    logging_interval: int = 1000
    inspection_goal =  0.1
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  True #not debug
    headless = True
    nav_camera_modality = "depth" # "rgb", "depth", or "rgbd"
    ptz_camera_modality = "depth" # "rgb", "depth", or "rgbd"

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
    # Treat sustained excessive chassis tilt as the same terminal safety event
    # as a confirmed contact crash.
    tipover_max_tilt_degrees: float = 45.0
    tipover_consecutive_steps: int = 2
    min_obstacles: int = 3
    max_obstacles: int = 15
    min_spawn_max_y: float = 5.0


    use_radius_aware_obstacle_spawning: bool = True
    robot_footprint_radius: float = 0.35
    fallback_target_footprint_radius: float = 0.8
    target_obstacle_surface_clearance: float = 1.4
    obstacle_obstacle_surface_clearance: float = 0.9
    robot_obstacle_surface_clearance: float = 1.2
    # Backward-compatible center-distance fallback for callers that do not
    # provide footprint radii.
    min_dist_between_obstacles: float = 2.2
    min_dist_to_objective: float = 2.0

    # Keep the pooling comparison policy-only; the shield would change the
    # transition distribution and confound the CLS-versus-mean ablation.
    use_directional_safety_shield: bool = False
    safety_shield_occupancy_threshold: float = 1.1
    safety_shield_robot_radius: float = 0.35
    safety_shield_margin: float = 0.10
    safety_shield_horizon_s: float = 0.50
    safety_shield_prediction_steps: int = 6
    safety_shield_z_min: float = 0.10
    safety_shield_z_max: float = 0.80
    safety_shield_linear_scales: tuple = (1.0, 0.75, 0.50, 0.25, 0.0)
class eval_Cfg:
    debug = False
    # Dedicated target semantics avoid collisions with authored ``class``
    # labels on the target's child meshes and elsewhere in the warehouse. Set
    # this temporarily to ``"class"`` only for old-evaluation compatibility.
    inspection_semantics_type: str = "inspection_target"
    # A semantic label alone can admit a face ID from a mismatched ray-caster
    # row in a vectorized evaluation. Require the active target mesh slot and
    # agreement between rendered depth and ray-cast depth at every counted
    # pixel. Set False to reproduce the legacy semantic-mask-only metric.
    use_strict_face_visibility_filter: bool = False
    face_depth_abs_tolerance_m: float = 0.05
    face_depth_rel_tolerance: float = 0.01
    # Must match the checkpoint architecture. Set False for the fixed-camera
    # ablation checkpoint and True for the active-PT models.
    enable_pt_actuation: bool = True
    # This must match the checkpoint architecture: False for four-action
    # baseline checkpoints, True for five-action zoom checkpoints.
    enable_camera_zoom: bool = False
    # Evaluation defaults to the original single-launch path. Enabling this is
    # architecture-neutral and can be useful for large vectorized evaluations.
    enable_map_ray_chunking: bool = True
    map_ray_chunk_percent: float = 25.0
    # Evaluate the August 31 policy on an out-of-distribution target using the
    # current physics, spawning, sensor noise, and collision detector.
    inspection_dataset = "evaluation"
    inspection_target = eval_dataset['ur10']
    # inspection_dataset = "primitive"
    # inspection_target = 'tessellated_thin_legged_body'
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

    max_episode_length: int = 1200
    min_episode_length: int = 1200
    logging_interval: int = 1500
    inspection_goal =  1.0
    visualisation_mode = None
    visualise_point_cloud = False # Only for debuggin the point cloud its incredinly memory intensive
    visualise_face_ids = False
    display_ray_counts = False
    display_cameras = False
    use_wandb =  False #not debug
    headless = True
    num_envs = 1 # Single environment for easier debugging
    nav_camera_modality = "depth"
    ptz_camera_modality = "depth" #rgbd
    use_depth_mask = False
    use_optical_flow_penalty = False
    use_optical_flow_as_quality = True
    fixed_spawns = False
    randomize_spawns = True
    use_hardest_curriculum = True
    start_crashes: int = 1
    end_crashes: int = 1
    collision_consecutive_steps: int = 2
    tipover_max_tilt_degrees: float = 45.0
    tipover_consecutive_steps: int = 2
    max_obstacles: int = 15
    reset_on_crash = True
    enable_voxel_visualization = False


    add_high_res_inspection_camera = True
    high_res_camera_width = 512
    high_res_camera_height = 512
    ignore_unattributed = False
    freeze_on_unattributed = False

    is_simplified = False

    use_radius_aware_obstacle_spawning: bool = True
    robot_footprint_radius: float = 0.35
    fallback_target_footprint_radius: float = 0.8
    target_obstacle_surface_clearance: float = 1.4
    obstacle_obstacle_surface_clearance: float = 0.9
    robot_obstacle_surface_clearance: float = 1.2
    # Backward-compatible center-distance fallback for callers that do not
    # provide footprint radii.
    min_dist_between_obstacles: float = 2.2
    min_dist_to_objective: float = 2.0

    # Keep this comparison policy-only. Otherwise the new shield would alter
    # the trajectory and confound the checkpoint comparison.
    use_directional_safety_shield: bool = False
    safety_shield_occupancy_threshold: float = 1.1
    safety_shield_robot_radius: float = 0.35
    safety_shield_margin: float = 0.10
    safety_shield_horizon_s: float = 0.50
    safety_shield_prediction_steps: int = 6
    safety_shield_z_min: float = 0.10
    safety_shield_z_max: float = 0.80
    safety_shield_linear_scales: tuple = (1.0, 0.75, 0.50, 0.25, 0.0)



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
mode_by_name = {
    "debug": modes[0],
    "train": modes[1],
    "eval": modes[2],
    "record": modes[3],
}
mode = 'eval'
requested_mode = os.environ.get("ISAACLAB_INSPECTION_RUN_MODE", mode).lower()
if requested_mode not in mode_by_name:
    raise ValueError(
        "ISAACLAB_INSPECTION_RUN_MODE must be one of "
        f"{tuple(mode_by_name)}, got {requested_mode!r}"
    )
cfg_mode = mode_by_name[requested_mode]

# inspection_null.py uses this process-local override for imported factory or
# workshop scenes, which already contain their own clutter. Normal training and
# evaluation processes never set it and therefore retain their configured data.
if os.environ.get("ISAACLAB_INSPECTION_DISABLE_PROCEDURAL_OBSTACLES", "0") == "1":
    cfg_mode.is_simplified = True
    cfg_mode.max_obstacles = 0

    # record_Cfg, #3





class record_Cfg:
    debug = False
    enable_pt_actuation: bool = True
    enable_camera_zoom: bool = False
    enable_map_ray_chunking: bool = False
    map_ray_chunk_percent: float = 12.5
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
    tipover_max_tilt_degrees: float = 45.0
    tipover_consecutive_steps: int = 2
