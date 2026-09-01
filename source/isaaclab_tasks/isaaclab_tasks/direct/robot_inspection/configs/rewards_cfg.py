from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    alpha = 0.5
    mesh_coverage_reward_scale: float =4e+2#0.2
    face_quality_k = 30 #60
    use_angle_weighted_reward: bool = True
    coverage_reward: float = 10.0
    # Map based rewards
    information_gain_reward_scale: float =    3e-5 # in expopnent    
    visibility_increase_reward_scale: float = 4e-4
    visitation_reward_scale: float = 5e-4
    # The occupancy map is an early-warning proxy, so keep it as a modest
    # additive shaping cost. A confirmed physical collision is handled below
    # with an exclusive terminal reward.
    occupancy_penalty_scale: float = 0.15   
    collision_threshold: float = 10.0
    terminal_collision_penalty: float = 1.0

    exploration_success_bonus: float = 2.0

    action_penalty_scale: float = 0.01 # original: 1e-5
    ptz_penalty_scale: float = 0.006 # original: 1e-5

    optical_flow_penalty_scale: float = 0.0 # 1e-3
    optical_flow_threshold: float = 12.5
    min_inspection_distance: float = 1.5

    # Penalties
    time_penalty: float = 1.0e-3 # Increased from 0.3e-3 to discourage wasting time in circles
    

    # visitation_reward_scale: float = 0.5e-3
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits
    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9
