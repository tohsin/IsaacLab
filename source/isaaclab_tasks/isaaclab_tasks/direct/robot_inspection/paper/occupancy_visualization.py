import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def create_occupancy_map():
    # Grid size - 9x9
    grid_size = 9
    
    # States
    UNKNOWN = 0.5
    FREE = 0.0
    OCCUPIED = 1.0
    
    # Initialize grid with UNKNOWN everywhere
    occupancy_grid = np.full((grid_size, grid_size), UNKNOWN)
    
    # Define explored region (Free space)
    # Robot at (4,4). Let's define a roughly circular/diamond explored area
    # Center 3x3 is definitely free
    for r in range(3, 6):
        for c in range(3, 6):
            occupancy_grid[r, c] = FREE
            
    # Extend exploration
    explored_extras = [
        (2, 4), (2, 3), (2, 5), (6, 4), (6, 3), (6, 5),
        (4, 2), (3, 2), (5, 2), (4, 6), (3, 6), (5, 6),
        (1, 4), # Looking at wall
        (7, 3), # Near random wall
        (7, 4),
        (5, 1)
    ]
    for r, c in explored_extras:
        occupancy_grid[r, c] = FREE

    # Define Walls (Occupied)
    # Matching Visibility Map walls
    # Surface: [(1, 3), (1, 4), (1, 5)]
    walls = [(1, 3), (1, 4), (1, 5)]
    # Random Wall: [(6, 1), (7, 1), (7, 2)]
    walls.extend([(6, 1), (7, 1), (7, 2)])
    
    for r, c in walls:
        occupancy_grid[r, c] = OCCUPIED
        
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Custom colormap
    # We have values 0.0, 0.5, 1.0
    # Map: 
    # 0.0 -> White (Free)
    # 0.5 -> Gray (Unknown)
    # 1.0 -> Black (Occupied)
    # We need a colormap that handles this.
    # Let's use a LinearSegmentedColormap or just manually map values if we used RGB.
    # But imshow with a ListedColormap is easier if we map values to integers 0, 1, 2.
    
    # Let's map for display:
    # 0 (Free) -> 0
    # 0.5 (Unknown) -> 1
    # 1.0 (Occupied) -> 2
    
    display_grid = np.zeros_like(occupancy_grid, dtype=int)
    # Default 0 (Free) is fine for Free.
    # Set Unknown
    display_grid[occupancy_grid == UNKNOWN] = 1
    # Set Occupied
    display_grid[occupancy_grid == OCCUPIED] = 2
    
    # Colors: [Light Yellow, Light Purple, Green]
    # Free=Light Yellow (#FFFACD), Unknown=Light Purple, Occupied=Green
    cmap = mcolors.ListedColormap(['#FFFACD', '#E8DAEF', '#2ecc71'])
    
    # Plot grid
    ax.imshow(display_grid, cmap=cmap, interpolation='nearest', origin='upper', vmin=0, vmax=2)
    
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

    # No Robot dot requested ("no need to show camera angle just show looks like an occuapncy map")
    # But usually nice to know where robot is? User said "ignore everything else outside your greed".
    # User said "no need to show camera angle".
    # I'll exclude robot dot to keep it purely a map.
    
    # Save
    output_path = "occupancy_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    create_occupancy_map()
