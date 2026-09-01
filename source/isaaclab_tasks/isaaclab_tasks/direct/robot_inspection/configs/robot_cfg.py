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
    pan_speed: float = 0.6  # Speed of the pan-tilt unit
    tilt_speed: float = 0.6  # Speed of the pan-tilt unit
    default_focal_length: float = 35.0  # Fixed inspection-camera focal length
    # optical flow parameters
    flow_safe_zone: float = 12.5
    flow_drop_speed: float = 12.0
