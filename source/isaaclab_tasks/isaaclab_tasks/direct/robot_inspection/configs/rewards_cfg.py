from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    alpha = 0.5
    mesh_coverage_reward_scale: float = 1.0
    face_quality_k = 60.0
    use_angle_weighted_reward: bool = True
    coverage_reward: float = 5.0
    information_gain_reward_scale: float =    0.3e-4 # in expopnent    
    visibility_increase_reward_scale: float = 4e-4
    exploration_success_bonus: float = 2.0

    action_penalty_scale: float = 1e-5
    ptz_penalty_scale: float = 1e-5 

    # Penalties
    time_penalty: float = 1e-3
    

    visitation_reward_scale: float = 0.5e-3
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits
    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9
