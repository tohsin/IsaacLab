linear_velocity  = 1.3
angular_velocity = 4 #1.0 * 4.0
wheel_separation = 	0.25 #  	0.37558 
wheel_radius = 0.098
left_wheel_velocity = (linear_velocity - (angular_velocity * wheel_separation / 2)) / wheel_radius
right_wheel_velocity = (linear_velocity + (angular_velocity * wheel_separation / 2)) / wheel_radius

print(f"Setting left wheel velocity to {left_wheel_velocity}")
print(f"Setting right wheel velocity to {right_wheel_velocity}")