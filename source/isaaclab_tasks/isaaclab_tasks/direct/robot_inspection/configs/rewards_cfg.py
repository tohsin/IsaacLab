from isaaclab.utils import configclass
from .mapping_cfg import RESOLUTION

@configclass
class RewardsCfg:
    """Configuration for all reward function terms."""
    resolution_ratio: float = (RESOLUTION / 0.25) ** 3
     
    alpha = 0.5
    mesh_coverage_reward_scale: float =5e+2#0.2
    face_quality_k = 60 #60
    use_angle_weighted_reward: bool = True
    coverage_reward: float = 5.0
    information_gain_reward_scale: float =    0.3e-4 * resolution_ratio # in expopnent    
    visibility_increase_reward_scale: float = 4e-4 * resolution_ratio
    exploration_success_bonus: float = 2.0

    action_penalty_scale: float = 0.3e-5 # original: 1e-5
    ptz_penalty_scale: float = 0.3e-5 # original: 1e-5

    optical_flow_penalty_scale: float = 0.0 # 1e-3
    optical_flow_threshold: float = 12.5

    # Penalties
    time_penalty: float = 0.3e-3 # original: 1e-3
    collision_penalty_scale: float = 2
    collision_threshold: float = 1.0
    

    visitation_reward_scale: float = 0.5e-3 * resolution_ratio
   
    # Visitation penalties to encourage exploration. a * e^N_of_visits
    visibility_decay_factor : float = 0.5
    visitation_decay_factor: float = 0.9

