from isaaclab.utils import configclass

@configclass
class RobotPhysicsCfg:
    """Parameters defining the robot's physical properties and limits."""
    wheel_separation: float = 0.37558
    wheel_radius: float = 0.095
    forward_vel: float = 6.0
    turn_vel: float = 9.0
    max_linear_velocity: float = 2.0  # 1.3 discrete
    max_angular_velocity: float = 4.0
    max_wheel_velocity: float = 20.0  # Max wheel velocity for the robot
    # PTZ Camera control configurations
    pan_speed: float = 0.7  # Speed of the pan-tilt unit
    tilt_speed: float = 0.7  # Speed of the pan-tilt unit
    # Zoom is intentionally discrete. Re-authoring many rendered USD cameras
    # every control step was unstable, so the environment applies at most a
    # small number of real focal-length changes per step.
    zoom_focal_lengths: tuple[float, ...] = (24.0, 35.0, 50.0)
    default_focal_length: float = 35.0
    zoom_action_hysteresis: float = 0.08
    max_zoom_updates_per_step: int = 32
    # optical flow parameters
    flow_safe_zone: float = 12.5
    flow_drop_speed: float = 12.0
