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


from a_star import a_star

def select_best_frontier(clusters, robot_pose_px, occupancy_grid=None, is_global=False, resolution=0.2, blacklist=None):
    """
    Selects the best frontier based on A* path distance.
    
    Args:
        clusters: List of frontiers (which are lists of pixel coordinates).
        robot_pose_px: The (x, y) pixel coordinates of the robot.
        occupancy_grid: 2D numpy array representing the map.
        is_global: Boolean indicating if it's a global map.
        resolution: Map resolution in meters/pixel.
        blacklist: List of (x, y) coordinates of blacklisted frontiers.
        
    Returns:
        The centroid (x, y) of the best frontier cluster, or None if no reachable frontiers.
    """
    if not clusters:
        return None
        
    if blacklist is None:
        blacklist = []
        
    best_dist = float('inf')
    best_centroid = None
    
    ry, rx = robot_pose_px
    
    obstacle_grid = None
    if occupancy_grid is not None:
        # Assuming values > 0.1 represent occupied space
        obstacle_grid = occupancy_grid > 0.1
    
    cluster_info = []
    for cluster in clusters:
        # Calculate centroid of the cluster
        y_coords = [p[0] for p in cluster]
        x_coords = [p[1] for p in cluster]
        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))
        centroid = (centroid_y, centroid_x)
        
        # Check against blacklist (3 pixel radius = 0.6m)
        is_blacklisted = False
        for by, bx in blacklist:
            if np.hypot(centroid_x - bx, centroid_y - by) < 3.0:
                is_blacklisted = True
                break
        
        if is_blacklisted:
            continue
            
        # Fallback/heuristic Euclidean distance
        euclid_dist = np.sqrt((centroid_x - rx)**2 + (centroid_y - ry)**2)
        cluster_info.append((euclid_dist, centroid))
        
    cluster_info.sort(key=lambda x: x[0])
    
    # Only A* the top 3 closest Euclidean frontiers to save CPU
    for euclid_dist, centroid in cluster_info[:3]:
        if obstacle_grid is not None:
            path = a_star(robot_pose_px, centroid, obstacle_grid)
            if path is None:
                continue # Path is unreachable
                
            # Calculate path length
            dist = 0.0
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i+1]
                dist += np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        else:
            dist = euclid_dist
        
        # Favor closer frontiers
        if dist < best_dist:
            best_dist = dist
            best_centroid = centroid
            
    return best_centroid
