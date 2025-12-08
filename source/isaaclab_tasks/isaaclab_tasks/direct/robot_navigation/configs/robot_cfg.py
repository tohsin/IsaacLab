from isaaclab.utils import configclass

@configclass
class RobotPhysicsCfg:
    """Parameters defining the robot's physical properties and limits."""
    wheel_separation: float = 0.25 # 0.37558
    wheel_radius: float = 0.098
    forward_vel: float = 6.0
    turn_vel: float = 9.0
    max_linear_velocity: float = 2.0  # 1.3 discrete
    max_angular_velocity: float = 4.0
    max_wheel_velocity: float = 18.0  # Max wheel velocity for the robot