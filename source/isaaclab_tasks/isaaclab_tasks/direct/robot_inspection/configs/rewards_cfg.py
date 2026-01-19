from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    alpha = 0.5
    mesh_coverage_reward_scale: float = 0.75
    coverage_reward: float = 5.0
    information_gain_reward_scale: float =   0.0001
    visibility_increase_reward_scale: float = 0.0001
    exploration_success_bonus: float = 2.0
    action_penalty_scale: float = -0.00005
    ptz_penalty_scale: float = -0.0001

    # Penalties
    time_penalty: float = -0.001
    

    visitation_reward_scale: float = 0.0075
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits
    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9
