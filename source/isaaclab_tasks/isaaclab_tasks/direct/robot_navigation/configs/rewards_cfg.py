from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""

    goal_reached_bonus: float = 10.0 # Sparse
 
    progress_reward_scale: float = 1.0 # dense
    # Penalties
    time_penalty: float = -0.001
    # Visitation penalties to encourage exploration. a * e^N_of_visits

