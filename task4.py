def hill_climbing(initial_state, elevations):
    """Runs the hill climbing algorithm and returns the index of the reached
    state"""
    current_state = initial_state
    
    # Keep track of the path taken
    path = [current_state]
    
    # Continue until no better neighboring state is found
    while True:
        # Get possible neighboring states (previous and next watch towers)
        neighbors = []
        
        # Add previous watch tower if it exists
        if current_state > 0:
            neighbors.append(current_state - 1)
        
        # Add next watch tower if it exists
        if current_state < len(elevations) - 1:
            neighbors.append(current_state + 1)
        
        # Find the neighbor with the highest elevation
        best_neighbor = current_state
        for neighbor in neighbors:
            if elevations[neighbor] > elevations[best_neighbor]:
                best_neighbor = neighbor
        
        # If no better neighbor is found, we've reached a local maximum
        if best_neighbor == current_state:
            break
        
        # Move to the best neighbor
        current_state = best_neighbor
        path.append(current_state)
    
    # Print the path taken for analysis
    print(f"Path taken: {path}")
    print(f"Elevations: {[elevations[i] for i in path]}")
    
    return current_state

# Run the algorithm
elevations = [500, 644, 713, 682, 560, 395, 249, 175, 199, 307, 449, 566, 607, 554, 427, 277, 162, 127, 184, 309]
initial_state = 16

final_state = hill_climbing(initial_state, elevations)
print(f'Final state: Watch tower {final_state}')
print(f'Elevation of final state: {elevations[final_state]}')

# Find the global maximum for comparison
global_max_state = elevations.index(max(elevations))
print(f'Global maximum: Watch tower {global_max_state} with elevation {elevations[global_max_state]}')