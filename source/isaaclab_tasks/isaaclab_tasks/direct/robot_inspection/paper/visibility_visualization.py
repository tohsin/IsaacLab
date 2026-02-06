import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge

def create_visibility_map():
    # Grid size - 9x9 for symmetry
    grid_size = 9
    
    # Initialize grid with white background everywhere
    # 0 = white, 1 = green
    visibility_grid = np.zeros((grid_size, grid_size))
    
    # Robot position - Egocentric center of 9x9 is (4, 4)
    robot_pos = (4, 4) 
    
    # Define surface (Green spots)
    # Original target surface (centered above robot)
    # Robot at row 4. Surface at row 1 (distance 3).
    # Cols centered at 4: 3, 4, 5
    surface_cells = [(1, 3), (1, 4), (1, 5)]
    
    # Add a "random wall" somewhere else
    # Let's put smooth wall fragment at bottom left area
    random_wall = [(6, 1), (7, 1), (7, 2)]
    
    all_green_cells = surface_cells + random_wall
    
    for r, c in all_green_cells:
        visibility_grid[r, c] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Custom colormap: 0=Light Yellow, 1=Green
    # Using #FFFACD (LemonChiffon)
    cmap = mcolors.ListedColormap(['#FFFACD', '#2ecc71'])
    
    # Plot grid
    ax.imshow(visibility_grid, cmap=cmap, interpolation='nearest', origin='upper', vmin=0, vmax=1)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    
    # Hide spines (borderless look)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw Field of View (Wedge)
    # Robot at (4, 4)
    # Surface at (1, 4) (center of surface)
    # Direction is Up (-y). 
    
    center = (robot_pos[1], robot_pos[0]) # (4, 4)
    radius = 4.8 # Bigger radius (reaches past the surface at distance 3)
    
    # Angle: Up is 270 (or -90)
    fov_angle = 80 # Bigger angle
    start_angle = 270 - (fov_angle/2)
    end_angle = 270 + (fov_angle/2)
    
    # Blue Wedge
    wedge = Wedge(center, radius, start_angle, end_angle, 
                  facecolor='#3498db', alpha=0.4, edgecolor=None)
    ax.add_patch(wedge)
    

    
    # Plot Robot dot
    ax.plot(robot_pos[1], robot_pos[0], marker='o', color='black', markersize=8, label='Robot')
    
    # Restore full grid view (remove slicing) and enforce exact limits
    # Grid goes from -0.5 to 8.5 (for 9 cells)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5) # Inverted Y for image
    
    # Ensure background is the same yellow, just in case of any gaps
    ax.set_facecolor('#FFFACD')
    fig.patch.set_facecolor('#FFFACD')

    # Save
    output_path = "visibility_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    create_visibility_map()
