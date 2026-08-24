import heapq
import numpy as np

def a_star(start, goal, obstacle_grid):
    """
    A* pathfinding on a 2D grid.
    
    Args:
        start: (x, y) tuple for start position.
        goal: (x, y) tuple for goal position.
        obstacle_grid: 2D boolean numpy array where True means obstacle.
        
    Returns:
        List of (x, y) tuples representing the path, or None if no path found.
    """
    shape_x, shape_y = obstacle_grid.shape
    
    if not (0 <= start[0] < shape_x and 0 <= start[1] < shape_y):
        return None
    if not (0 <= goal[0] < shape_x and 0 <= goal[1] < shape_y):
        return None
        
    if obstacle_grid[start[0], start[1]]:
        # Start is in an obstacle, but maybe the robot is slightly clipping. Let's try to find a path anyway.
        # But technically A* needs a valid start. Let's allow it but warn or just continue.
        pass
    #8 neighbor
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    def heuristic(a, b):
        return np.hypot(b[0] - a[0], b[1] - a[1])
        
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    
    # Optional: limit search space to avoid freezing on impossible paths
    max_iters = 50000
    iters = 0
    
    while open_set:
        iters += 1
        if iters > max_iters:
            return None # Give up if it takes too long
            
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if not (0 <= neighbor[0] < shape_x and 0 <= neighbor[1] < shape_y):
                continue
                
            # If the neighbor is an obstacle and it's not the goal
            # (Goal might be a frontier cell which could be technically slightly occupied/unknown)
            if neighbor != goal and obstacle_grid[neighbor[0], neighbor[1]]:
                continue
                
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g_score = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
    return None