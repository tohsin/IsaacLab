lin_cmd = 1
ang_cmd = 1

linear_velocity  = lin_cmd * 1.0
angular_velocity = ang_cmd * 3 #1.0 * 4.0
wheel_separation = 	0.37558 # 0.25 	0.37558 
wheel_radius = 0.095
left_wheel_velocity = (linear_velocity - (angular_velocity * wheel_separation / 2)) / wheel_radius
right_wheel_velocity = (linear_velocity + (angular_velocity * wheel_separation / 2)) / wheel_radius

print(f"Setting left wheel velocity to {left_wheel_velocity}")
print(f"Setting right wheel velocity to {right_wheel_velocity}")