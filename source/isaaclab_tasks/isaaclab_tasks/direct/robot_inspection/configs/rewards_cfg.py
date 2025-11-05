from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    mesh_coverage_reward_scale: float = 0.01
    coverage_reward: float = 3.0
    information_gain_reward_scale: float = 0.001
    exploration_success_bonus: float = 2.0

    # Penalties
    time_penalty: float = -0.001

    visibility_increase_reward_scale: float = 0.003
    visitation_reward_scale: float = 0.0075
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits

    

    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9
