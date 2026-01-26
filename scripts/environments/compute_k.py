import numpy as np
def compute_k( N_good, q):
    A = N_good 
    denominator = np.log(1 - q)
    return -1 * A/denominator

print(compute_k(N_good= 200, q =0.02))

def compute_N(k, q):
    return -1 * k * np.log(1 - q)

print(compute_N(k = 90, q=0.02))