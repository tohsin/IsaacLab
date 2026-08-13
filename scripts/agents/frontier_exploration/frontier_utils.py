import numpy as np
import cv2
import scipy.ndimage as ndimage

def find_frontiers(occupancy_grid: np.ndarray, free_thresh: float = 0.2, unknown_thresh: float = 0.5) -> np.ndarray:
    """
    Finds frontier cells in an occupancy grid.
    
    Args:
        occupancy_grid: 2D numpy array representing the map (e.g. 0.0=free, 1.0=occupied, 0.5=unknown).
        free_thresh: Values below this are considered free space.
        unknown_thresh: Values around this are considered unknown.
        
    Returns:
        A boolean mask of the same shape as occupancy_grid, True where frontiers exist.
    """
    # Create masks for free and unknown space
    free_space = occupancy_grid < free_thresh
    
    # Assuming unknown space is around 0.5, or you can just use anything not free and not occupied
    # For a log-odds map, 0 might be unknown, negatives free, positives occupied.
    # Adjust according to your specific map representation.
    unknown_space = (occupancy_grid >= free_thresh) & (occupancy_grid <= unknown_thresh)
    
    # A frontier is an unknown cell that is adjacent to a free cell
    # We can find this by dilating the free space and finding the intersection with unknown space
    structuring_element = ndimage.generate_binary_structure(2, 2) # 8-connected
    dilated_free = ndimage.binary_dilation(free_space, structure=structuring_element)
    
    frontiers = dilated_free & unknown_space
    return frontiers


def cluster_frontiers(frontiers_mask: np.ndarray) -> list:
    """
    Clusters adjacent frontier cells into connected components.
    
    Args:
        frontiers_mask: 2D boolean numpy array.
        
    Returns:
        List of lists, where each inner list contains (y, x) tuples for cells in one cluster.
    """
    # Convert mask to uint8 for cv2
    img = (frontiers_mask * 255).astype(np.uint8)
    
    # Find contours or connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    
    clusters = []
    # Start from 1 to ignore the background (label 0)
    for label in range(1, num_labels):
        # Find coordinates of all pixels in this component
        y_coords, x_coords = np.where(labels == label)
        cluster = list(zip(y_coords, x_coords))
        clusters.append(cluster)
        
    return clusters


def select_best_frontier(clusters: list, robot_pose_px: tuple) -> tuple:
    """
    Selects the best frontier cluster based on a simple heuristic (e.g., closest distance to robot).
    
    Args:
        clusters: List of frontier clusters (list of (y, x) coordinates).
        robot_pose_px: The robot's position in grid coordinates (y, x).
        
    Returns:
        The centroid (y, x) of the best frontier cluster, or None if no frontiers.
    """
    if not clusters:
        return None
        
    best_dist = float('inf')
    best_centroid = None
    
    ry, rx = robot_pose_px
    
    for cluster in clusters:
        # Calculate centroid of the cluster
        y_coords = [p[0] for p in cluster]
        x_coords = [p[1] for p in cluster]
        
        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))
        
        # Calculate distance to robot
        dist = np.sqrt((centroid_y - ry)**2 + (centroid_x - rx)**2)
        
        # Favor closer frontiers. (You can also incorporate cluster size into the heuristic).
        if dist < best_dist:
            best_dist = dist
            best_centroid = (centroid_y, centroid_x)
            
    return best_centroid
