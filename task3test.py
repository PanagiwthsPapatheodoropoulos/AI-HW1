import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.distance import lonlat, distance

# Load world map
worldmap = gpd.read_file("https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson")

# ...existing code...

# Load distance data from file
def load_distances(filename):
    """Load distances from a CSV file"""
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Get city names from the first line
    cities = lines[0].strip().split(',')
    
    # Create empty distance matrix
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities), dtype=int)
    
    # Fill the distance matrix
    for i in range(1, n_cities+1):
        row = lines[i].strip().split(',')
        for j in range(1, n_cities+1):
            distances[i-1, j-1] = int(row[j]) if row[j] else 0
    
    return distances, cities

def route_length(chromosome, distances):
    """Calculate the total length of a route"""
    length = 0
    for i in range(len(chromosome)-1):
        length += distances[chromosome[i], chromosome[i+1]]
    return length

def fitness(chromosome, distances):
    """Returns the fitness of a chromosome. For TSP, we want to minimize the route length,
    so fitness is inversely proportional to route length."""
    length = route_length(chromosome, distances)
    return 1 / length  # Higher fitness means shorter route

def init_population(n_chromosomes, n_cities):
    """Initializes and returns the population of chromosomes for TSP"""
    population = []
    for _ in range(n_chromosomes):
        # Create a permutation of cities 1-21
        middle_cities = list(range(1, n_cities))
        random.shuffle(middle_cities)
        # Add Athens (0) at beginning and end
        chromosome = [0] + middle_cities + [0]
        population.append(chromosome)
    return population

def select(population, fitness_values):
    """Tournament selection: randomly selects two chromosomes with probability 
    proportional to their fitness value."""
    tournament_size = 3
    
    # First parent
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    chromosome_1 = population[winner_idx]
    
    # Second parent
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    chromosome_2 = population[winner_idx]
    
    return chromosome_1, chromosome_2

def crossover(chromosome1, chromosome2):
    """Implements ordered crossover for TSP, ensuring legal chromosomes are produced.
    This method preserves the relative order of cities from each parent while
    ensuring no city is repeated and Athens is at the start and end."""
    # We keep the starting and ending city (Athens) fixed
    size = len(chromosome1)
    
    # Select a segment to copy from parent 1 (excluding start/end points)
    point1 = random.randint(1, size - 3)
    point2 = random.randint(point1, size - 2)
    
    # Initialize offspring with placeholders (-1 means "to be filled")
    offspring = [-1] * size
    
    # Fix start and end as Athens (0)
    offspring[0] = 0
    offspring[-1] = 0
    
    # Copy the selected segment from parent 1
    for i in range(point1, point2 + 1):
        offspring[i] = chromosome1[i]
    
    # Create a list of cities that are already in the offspring
    used_cities = set(offspring)
    
    # Fill the remaining positions with cities from parent 2 in order
    # but skipping cities already in the offspring
    j = 1  # Start after Athens
    for i in range(1, size - 1):
        if offspring[i] == -1:  # If position is empty
            # Find the next city from parent 2 that's not already in offspring
            while True:
                if j >= size - 1:
                    j = 1  # Reset if we reached the end (excluding Athens at end)
                
                if chromosome2[j] not in used_cities:
                    offspring[i] = chromosome2[j]
                    used_cities.add(chromosome2[j])
                    break
                j += 1
    
    return offspring

def mutate(chromosome, prob_mutation):
    """Applies mutation to the chromosome by swapping two cities (excluding Athens)"""
    if random.random() < prob_mutation:
        # Only swap cities in positions 1 to len-2 (keeping Athens at start and end)
        idx1 = random.randint(1, len(chromosome) - 2)
        idx2 = random.randint(1, len(chromosome) - 2)
        while idx1 == idx2:
            idx2 = random.randint(1, len(chromosome) - 2)
        
        # Swap the cities
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    
    return chromosome

def genetic_algorithm(population, distances, n_generations, prob_mutation):
    """Implements the genetic algorithm for TSP"""
    n_population = len(population)
    best_solution = None
    best_fitness = -float('inf')
    best_distance = float('inf')
    
    # Store history for plotting
    history = []
    
    for generation in range(n_generations):
        # Calculate fitness for each chromosome
        fitness_values = [fitness(chromosome, distances) for chromosome in population]
        
        # Find the best solution in current generation
        current_best_idx = fitness_values.index(max(fitness_values))
        current_best = population[current_best_idx]
        current_best_fitness = fitness_values[current_best_idx]
        current_distance = route_length(current_best, distances)
        
        # Update overall best if needed
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best.copy()
            best_distance = current_distance
        
        # Store history
        history.append(current_distance)
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best distance = {best_distance} km")
        
        # Create new population
        new_population = []
        
        # Elitism: preserve the best chromosome
        new_population.append(current_best)
        
        # Generate new chromosomes
        while len(new_population) < n_population:
            # Selection
            parent1, parent2 = select(population, fitness_values)
            
            # Crossover
            offspring = crossover(parent1, parent2)
            
            # Mutation
            offspring = mutate(offspring, prob_mutation)
            
            # Add to new population
            new_population.append(offspring)
        
        # Replace old population
        population = new_population
    
    print(f"Final best distance: {best_distance} km")
    return best_solution, history

def plot_route(solution, df, worldmap):
    """Plot the route on a world map"""
    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(16, 10))
    worldmap.plot(color="lightgrey", ax=ax)
    
    # Create GeoDataFrame from city coordinates
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    
    # Plot cities
    gdf.plot(ax=ax, color="red", markersize=50)
    
    # Plot the route
    city_lonlat = df[["Longitude", "Latitude"]].to_numpy()
    route_x = city_lonlat[solution, 0]
    route_y = city_lonlat[solution, 1]
    ax.plot(route_x, route_y, 'b-', linewidth=1.5)
    
    # Add city labels
    for idx, row in df.iterrows():
        ax.annotate(row['City'], xy=(row['Longitude'], row['Latitude']), 
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=8, color='black', fontweight='bold')
    
    # Add title and save
    plt.title("Optimized Global Travel Route (Starting and Ending in Athens)")
    plt.savefig("optimized_route.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence(history):
    """Plot the convergence of the algorithm"""
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Convergence of Genetic Algorithm')
    plt.xlabel('Generation')
    plt.ylabel('Best Route Length (km)')
    plt.grid(True)
    plt.savefig("convergence.png", dpi=300)
    plt.show()

def run_tsp_solution(distances_file=None, city_data=None, distances=None):
    """Run the complete TSP solution with the genetic algorithm"""
    if distances_file:
        # Load distances from file
        distances, cities = load_distances(distances_file)
    
    # Algorithm parameters
    population_size = 100
    n_generations = 500
    prob_mutation = 0.2
    n_cities = distances.shape[0]
    
    # Initialize population
    population = init_population(population_size, n_cities)
    
    # Run genetic algorithm
    best_solution, history = genetic_algorithm(
        population, 
        distances, 
        n_generations=n_generations, 
        prob_mutation=prob_mutation
    )
    
    # Print the final route
    if city_data is not None:
        route_cities = [city_data['City'][i] for i in best_solution]
        print("\nOptimal Route:")
        for i, city in enumerate(route_cities):
            print(f"{i+1}. {city}")
    
    # Plot convergence
    plot_convergence(history)
    
    # Plot route on map if city data is available
    if city_data is not None and 'worldmap' in globals():
        plot_route(best_solution, city_data, worldmap)
    
    return best_solution

# Example usage
if __name__ == "__main__":
    # Using the provided data
    # Note: This would be replaced with actual file loading in real execution
    df = pd.DataFrame({
        "City": ["Athens", "Sydney", "Moscow", "Casablanca", "San Fransisco", "New York", 
                "Toronto", "Addis Ababa", "Helsinki", "Tokyo", "Hà Noi", "Abu Dhabi", 
                "Bangkok", "Paris", "Victoria", "Lima", "Mexico City", "Beijing", 
                "Cairo", "Buenos Aires", "Santiago", "Caracas"],
        "Country": ["Greece", "Australia", "Russia", "Morocco", "USA", "USA", "Canada", 
                    "Ethiopia", "Finland", "Japan", "Vietnam", "United Arab Emirates", 
                    "Thailand", "France", "Seychelles", "Peru", "Mexico", "China", 
                    "Egypt", "Argentina", "Chile", "Venezuela"],
        "Latitude": [37.95, -33.86, 55.75, 33.58, 37.77, 40.71, 43.70, 9.02, 60.16, 
                    35.68, 21.02, 24.46, 13.72, 48.85, -4.61, -12.04, 19.42, 39.90, 
                    30.03, -34.58, -33.45, 10.48],
        "Longitude": [23.74, 151.20, 37.62, -7.61, -122.41, -74.00, -79.41, 38.74, 
                    24.94, 139.69, 105.84, 54.36, 100.52, 2.34, 55.45, -77.02, 
                    -99.14, 116.39, 31.23, -58.66, -70.66, -66.86],
    })
    
    # Calculate distances if not provided
    if 'distances' not in globals():
        distances = np.zeros((22, 22), dtype=int)
        for index1, row1 in df.iterrows():
            for index2, row2 in df.iterrows():
                distances[index1, index2] = int(distance(lonlat(row1['Longitude'], row1['Latitude']),  lonlat(row2['Longitude'], row2['Latitude'])).km)
    
    # Run the solution
    best_solution = run_tsp_solution(distances=distances, city_data=df)