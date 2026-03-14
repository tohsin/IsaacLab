import numpy as np
from matplotlib import pyplot as plt

def compute_flow_behavior(sigma, flow_values):
    # Using the original Gaussian decay
    return np.exp(-0.5 * np.square(flow_values / sigma))

def compute_flow_behavior_offset(safe_zone, drop_speed, flow_values):
    # Offset curve: Stays at 1.0 until safe_zone, then drops
    active_penalty = np.maximum(0, flow_values - safe_zone)
    return np.exp(-0.5 * np.square(active_penalty / drop_speed))

if __name__ == "__main__":
    flow_values = np.linspace(0, 50, 1000)
    
    # The existing curve you noticed was too aggressive
    sigma = 12.5 
    plt.plot(flow_values, compute_flow_behavior(sigma, flow_values), label=f"Original (Sigma={sigma})")
    
    # A new offset curve that tolerates movement up to 10.0 perfectly
    safe_zone = 10.0
    drop_speed = 10.0
    plt.plot(flow_values, compute_flow_behavior_offset(safe_zone, drop_speed, flow_values), 
             label=f"Offset (Tolerate up to {safe_zone}, then drop)", linewidth=2.5)
    
    plt.title("Optical Flow Quality Mask vs. Raw Flow Magnitude")
    plt.xlabel("Optical Flow Magnitude")
    plt.ylabel("Reward Multiplier (1.0 = Full Reward, 0.0 = No Reward)")
    plt.grid(True)
    plt.legend()
    plt.show()
    