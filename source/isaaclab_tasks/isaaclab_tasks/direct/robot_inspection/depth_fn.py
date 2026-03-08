import numpy as np
from matplotlib import pyplot as plt

def compute_depth_behavior(std, max_req_depth , depth_values):
    delta_d = np.minimum(0, (depth_values - max_req_depth))
    return np.exp(-0.5 * (delta_d / std) ** 2)


if __name__ == "__main__":
    std = 0.5
    max_req_depth = 1.0
    depth_values = np.linspace(0, 5, 1000)
    plt.plot(depth_values, compute_depth_behavior(std, max_req_depth, depth_values))
    plt.show()
    