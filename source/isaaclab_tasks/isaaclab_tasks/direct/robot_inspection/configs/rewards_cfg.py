from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    alpha = 0.5
    mesh_coverage_reward_scale: float = 0.2
    face_quality_k = 60 #60
    use_angle_weighted_reward: bool = True
    coverage_reward: float = 5.0
    information_gain_reward_scale: float = 3.9e-6 # mathematically scaled from 0.3e-4 due to 0.1m voxel map resolution  
    visibility_increase_reward_scale: float = 4.5e-5 # mathematically scaled from 4e-4 due to 0.1m voxel map resolution
    exploration_success_bonus: float = 2.0

    action_penalty_scale: float = 0.3e-5 # original: 1e-5
    ptz_penalty_scale: float = 0.3e-5 # original: 1e-5
    optical_flow_penalty_scale: float = 1e-2 # 1e-3
    optical_flow_threshold: float = 12.5

    # Penalties
    time_penalty: float = 0.3e-3 # original: 1e-3
    

    visitation_reward_scale: float =6.4e-5 # mathematically scaled from 0.5e-3 due to 0.1m voxel map resolution
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits
    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9
    
    # Depth-Based Reward Params
    max_inspection_distance: float = 1.0
    depth_sigma: float = 0.5   # Width of the "sweet spot"
