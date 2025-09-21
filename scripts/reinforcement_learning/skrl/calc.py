import numpy as np

R  = [[ 0.6533,  0.6403,  0.4040],
      [-0.3488,  0.7282, -0.5900],
      [-0.6719,  0.2445,  0.6991]]
R = np.array(R)
R_T = R.T

a = R @ R_T
print(a)
print(np.linalg.det(R))

R_11 = R[0][0]
R_22 = R[1][1]
R_33 = R[2][2]
R_13 = R[0][2]
R_23 = R[1][2]
R_21 = R[1][0]
R_12 = R[0][1]
R_32 = R[2][1]
R_31 = R[2][0]

print("R_13", R_13)
print("R_23", R_23)
print("R_33", R_33)
print("R_21", R_21)
print("R_12", R_12)
print("R_32", R_32)
print("R_31", R_31)
left_ = np.sqrt(R_13**2 + R_23**2)
theta =np.arctan2(left_, R_33)
print(np.degrees(theta))


phi = np.arctan2(R_23/ np.sin(theta), R_13/ np.sin(theta))
print("Postive Phi", np.degrees(phi))

phi = np.arctan2(R_23/ -np.sin(theta), R_13/ -np.sin(theta))
print("Negative Phi", np.degrees(phi))


psi = np.arctan2(R_32/ np.sin(theta), -R_31/ np.sin(theta))
print("Postive Psi", np.degrees(psi))
psi = np.arctan2(R_32/ -np.sin(theta), -R_31/ -np.sin(theta))
print("Negative Psi", np.degrees(psi))

# using angle aaxis
#cosine = (np.trace(R) - 1) / 2
cos_thata = (np.trace(R) - 1) / 2
print("cosine", cos_thata)

# inside multiple
inside = (R_32 - R_23)**2 + (R_13 - R_31)**2 + (R_21 - R_12)**2
print("inside", inside)
S_THETA = np.sqrt(inside) / 2
print("S_THETA", S_THETA)



# computing k
k_matrix = np.array([[R_32 - R_23],
                     [R_13 - R_31],
                     [R_21 - R_12]])
print("k_matrix", k_matrix)
k = 1/(2 *(S_THETA)) * k_matrix
print("k", k)

theta_ = np.arctan2(S_THETA, cos_thata)
print("Theta from axis angle", np.degrees(theta_))

# quaterions calcualtion 
q_0_sq = (1 + R_11 + R_22 + R_33 ) / 4
print("q_0_sq", q_0_sq)
print("q_0", q_0_sq**0.5)

q_1_sq = (1 + R_11 - R_22 - R_33 ) / 4
print("q_1_sq", q_1_sq)
print("q_1", q_1_sq**0.5)

q_2_sq = (1 - R_11 + R_22 - R_33 ) / 4
print("q_2_sq", q_2_sq)
print("q_2", q_2_sq**0.5)
q_3_sq = (1 - R_11 - R_22 + R_33 ) / 4
print("q_3_sq", q_3_sq)
print("q_3", q_3_sq**0.5)


q0q1 = (R_32 - R_23) / 4
print("q0q1", q0q1)

q0q2 = (R_13 - R_31) / 4
print("q0q2", q0q2)

q0q3 = (R_21 - R_12) / 4
print("q0q3", q0q3)

q1q2 = (R_12 + R_21) / 4
print("q1q2", q1q2)
q1q3 = (R_31 + R_13) / 4
print("q1q3", q1q3)
q2q3 = (R_23 + R_32) / 4
print("q2q3", q2q3)
