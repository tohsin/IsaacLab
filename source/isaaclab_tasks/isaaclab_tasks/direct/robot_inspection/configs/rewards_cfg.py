from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    mesh_coverage_reward_scale: float = 0.01
    coverage_reward: float = 3.0
    information_gain_reward_scale: float = 0.04

    # Penalties
    time_penalty: float = -0.001

    visibility_increase_reward_scale: float = 1.0
    visibility_decay_factor : float = 0.5
    # Visitation penalties to encourage exploration. a * e^N_of_visits

    visitation_reward_scale: float = 0.05 
    visitation_decay_factor: float = 0.8
