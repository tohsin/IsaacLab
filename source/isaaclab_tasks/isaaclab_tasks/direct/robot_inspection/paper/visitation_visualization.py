import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def create_visitation_map():
    # Grid size
    grid_size = 9
    
    # Initialize visitation grid (0 counts initially)
    # Using float to allow for normalization if needed, but integers are fine for counts
    visitation_grid = np.zeros((grid_size, grid_size))
    
    # Define a trajectory from top (row 0) to center (row 4)
    # Top-middle is (0, 4)
    # Center is (4, 4)
    
    # Simple path points (row, col)
    path = [
        (0, 4), (1, 4), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4)
    ]
    
    # Simulate visits
    # We'll add some "noise" or extra visits to make it look realistic
    # Higher counts near the path
    for r, c in path:
        visitation_grid[r, c] += 5  # Main path has high visits
        
        # Add some spillover to neighbors
        for nr, nc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                visitation_grid[nr, nc] += np.random.randint(1, 3)

    # Make the start and end most visited
    visitation_grid[0, 4] += 3
    visitation_grid[4, 4] += 5

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Custom colormap: Light Yellow to Dark Blue
    # Create a custom colormap that starts at light yellow
    # Using #FFFACD (LemonChiffon) which is more visible than LightYellow
    train_cmap = mcolors.LinearSegmentedColormap.from_list("YellowBlue", ["#FFFACD", "blue"])

    # Plot heatmap
    # Using 'nearest' interpolation for clear grid cells
    cax = ax.imshow(visitation_grid, cmap=train_cmap, interpolation='nearest', origin='upper', vmin=0)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    
    # Remove major ticks labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis ticks (the small lines)
    ax.tick_params(axis='both', which='both', length=0)

    # Annotate counts (optional, maybe clutter for 10x10 but let's see)
    for i in range(grid_size):
        for j in range(grid_size):
            if visitation_grid[i, j] > 0:
                text = ax.text(j, i, int(visitation_grid[i, j]),
                               ha="center", va="center", color="black" if visitation_grid[i,j] < 10 else "white", fontsize=8)

    # Plot Trajectory arrow
    # Extract y (rows) and x (cols)
    path_y = [p[0] for p in path]
    path_x = [p[1] for p in path]
    
    # Plot line
    ax.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=4, label='Trajectory')
    
    # Remove title
    # ax.set_title("Robot Visitation & Trajectory (10x10 Grid)")
    
    # Don't turn off axis completely, or we lose the grid
    # ax.axis('off')
    
    # Hide spines (the box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Ensure ticks are hidden but grid settings remain
    ax.set_xticks([])
    ax.set_yticks([])

    # Save
    output_path = "visitation_trajectory.png"
    # bbox_inches='tight', pad_inches=0 removes the white border
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    create_visitation_map()
