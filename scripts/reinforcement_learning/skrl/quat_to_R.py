q_0 = 1
q_1 = 0
q_2 = 0
q_3 = 0
import numpy as np
q = np.array([q_0, q_1, q_2, q_3])

def quat_to_R(q):
    q_0, q_1, q_2, q_3 = q
    R = np.array([[1 - (2*q_2**2) + (2*q_3**2), 2*(q_1*q_2 - q_0*q_3), 2*(q_1*q_3 + q_0*q_2)],
                   [2*(q_1*q_2 + q_0*q_3), q_0**2 - q_1**2 + q_2**2 - q_3**2, 2*(q_2*q_3 - q_0*q_1)],
                   [2*(q_1*q_3 - q_0*q_2), 2*(q_2*q_3 + q_0*q_1), q_0**2 - q_1**2 - q_2**2 + q_3**2]])
    return R
R = quat_to_R(q)
print("R", R)