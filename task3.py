import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from geopy.distance import lonlat, distance
from math import cos, pi

# For TSP, fitness is the inverse of route length (shorter routes are better)
def fitness(chromosome, distances):
    """Returns the fitness of a chromosome."""
    route_len = route_length(chromosome, distances)
    # Using inverse of route length so higher value = better fitness
    fitness_val = 1.0 / route_len
    return fitness_val

def init_population(n_chromosomes, n_cities):
    """Initializes and returns the population of chromosomes for TSP"""
    population = []
    for i in range(n_chromosomes):
        # Create a valid TSP route: start with Athens (0), end with Athens (0)
        # and visit all other cities exactly once
        middle_cities = list(range(1, n_cities))
        random.shuffle(middle_cities)
        chromosome = [0] + middle_cities + [0]
        population.append(chromosome)
    return population

def select(population, distances):
    """Randomly selects a pair of chromosomes with probability proportional to
    their fitness value."""
    # Calculate fitness for entire population
    population_fitness = [fitness(chromosome, distances) for chromosome in population]
    
    # Calculate total fitness
    total_fitness = sum(population_fitness)
    
    # Create normalized probabilities
    probabilities = [f/total_fitness for f in population_fitness]
    
    # Select two chromosomes based on fitness probability
    chromosome_indices = np.random.choice(
        range(len(population)), 
        size=2, 
        replace=False, 
        p=probabilities
    )
    
    return population[chromosome_indices[0]], population[chromosome_indices[1]]

def crossover(parent1, parent2):
    """Implements a crossover operator that maintains valid TSP tours.
    Always returns a valid chromosome where:
    - First and last positions are 0 (Athens)
    - All other cities appear exactly once"""
    
    # Length of chromosome (including both Athens positions)
    n = len(parent1)
    
    # Generate two random crossover points (excluding first and last positions)
    crossover_points = sorted(random.sample(range(1, n-1), 2))
    start, end = crossover_points
    
    # Initialize offspring with Athens at start and end
    offspring = [0] + [-1] * (n-2) + [0]
    
    # Copy segment from parent1
    for i in range(start, end+1):
        offspring[i] = parent1[i]
    
    # Get remaining cities from parent2 in order
    remaining_positions = [i for i in range(1, n-1) if i < start or i > end]
    remaining_cities = [city for city in parent2[1:-1] if city not in offspring]
    
    # Fill remaining positions
    for pos, city in zip(remaining_positions, remaining_cities):
        offspring[pos] = city
    
    return offspring

def mutate(chromosome, prob_mutation):
    """Applies mutation to the chromosome by swapping random cities.
    Preserves Athens at first and last positions."""
    mutated = chromosome.copy()
    
    # For each position (excluding first and last), decide whether to mutate
    for i in range(1, len(chromosome)-1):
        if random.random() < prob_mutation:
            # Find another position to swap with (excluding first and last)
            j = random.randint(1, len(chromosome)-2)
            if i != j:
                mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def genetic_algorithm(population, distances, n_generations, prob_mutation):
    """Implements the genetic algorithm. Returns the fittest chromosome of the
    last generation"""
    
    best_chromosome = None
    best_distance = float('inf')
    
    # Track best solutions for plotting
    best_distances = []
    
    for gen in range(n_generations):
        new_population = []
        
        # Elitism: keep the best chromosome
        if best_chromosome is not None:
            new_population.append(best_chromosome)
        
        # Create the rest of the population
        while len(new_population) < len(population):
            # Selection
            parent1, parent2 = select(population, distances)
            
            # Crossover
            offspring = crossover(parent1, parent2)
            
            # Mutation
            offspring = mutate(offspring, prob_mutation)
            
            # Add to new population
            new_population.append(offspring)
        
        # Update population
        population = new_population
        
        # Find the best chromosome in current generation
        for chromosome in population:
            distance = route_length(chromosome, distances)
            if distance < best_distance:
                best_distance = distance
                best_chromosome = chromosome
        
        best_distances.append(best_distance)
        
        # Optional: Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best distance = {best_distance}")
    
    # Plot the evolution of best distance
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_generations), best_distances)
    plt.title('Evolution of Best Distance over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance (km)')
    plt.grid(True)
    plt.show()
    
    return best_chromosome

def route_length(chromosome, distances):
    """Calculate the total length of a route"""
    length = 0
    for i in range(len(chromosome)-1):
        length += distances[chromosome[i], chromosome[i+1]]
    return length

# Main execution
df = pd.DataFrame(
    {
        "City": ["Athens", "Sydney", "Moscow", "Casablanca", "San Fransisco", "New York", "Toronto", "Addis Ababa", "Helsinki", "Tokyo", "Hà Noi", "Abu Dhabi", "Bangkok", "Paris", "Victoria", "Lima", "Mexico City", "Beijing", "Cairo", "Buenos Aires", "Santiago", "Caracas"],
        "Country": ["Greece", "Australia", "Russia", "Morocco", "USA", "USA", "Canada", "Ethiopia", "Finland", "Japan", "Vietnam", "United Arab Emirates", "Thailand", "France", "Seychelles", "Peru", "Mexico", "China", "Egypt", "Argentina", "Chile", "Venezuela"],
        "Latitude": [37.95, -33.86, 55.75, 33.58, 37.77, 40.71, 43.70, 9.02, 60.16, 35.68, 21.02, 24.46, 13.72, 48.85, -4.61, -12.04, 19.42, 39.90, 30.03, -34.58, -33.45, 10.48],
        "Longitude": [23.74, 151.20, 37.62, -7.61, -122.41, -74.00, -79.41, 38.74, 24.94, 139.69, 105.84, 54.36, 100.52, 2.34, 55.45, -77.02, -99.14, 116.39, 31.23, -58.66, -70.66, -66.86],
    }
)

# Calculate distances between cities
distances = np.zeros((22, 22), dtype=int)
for index1, row1 in df.iterrows():
    for index2, row2 in df.iterrows():
        distances[index1, index2] = int(distance(lonlat(row1['Longitude'], row1['Latitude']), lonlat(row2['Longitude'], row2['Latitude'])).km)

# Create geodataframe for plotting
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")

# Load world map
worldmap = gpd.read_file("https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson")

# Test with random chromosome
random_chromosome = [0, 5, 9, 1, 20, 14, 7, 19, 8, 12, 13, 21, 2, 18, 3, 16, 6, 11, 4, 10, 15, 17, 0]
print('Route length of random chromosome:', route_length(random_chromosome, distances))

# Plot random route
fig, ax = plt.subplots(figsize=(16, 10))
worldmap.plot(color="lightgrey", ax=ax)
gdf.plot(ax=ax, color="red")
city_lonlat = df[["Longitude", "Latitude"]].to_numpy()
plt.plot(city_lonlat[random_chromosome, 0], city_lonlat[random_chromosome, 1])
plt.title("Random Route")
plt.show()

# Run genetic algorithm
population_size = 100
n_cities = len(df)
n_generations = 300
prob_mutation = 0.1

# Initialize population
population = init_population(population_size, n_cities)

# Run genetic algorithm
solution = genetic_algorithm(population, distances, n_generations=n_generations, prob_mutation=prob_mutation)

print('Value of fittest chromosome:', route_length(solution, distances))
print('Route:', [df['City'][i] for i in solution])

# Plot optimal route
fig, ax = plt.subplots(figsize=(16, 10))
worldmap.plot(color="lightgrey", ax=ax)
gdf.plot(ax=ax, color="red")
plt.plot(city_lonlat[solution, 0], city_lonlat[solution, 1])
plt.title("Optimized Route using Genetic Algorithm")
plt.show()