import numpy as np
import matplotlib.pyplot as plt

# Vectorized implementation of the objective function
def f(population):
    """Compute function f for entire population at once."""
    return 100 + np.sum(population**2 - 10 * np.cos(2 * np.pi * population), axis=1)

def fitness(chromosome):
    """
    Calculate fitness for a single chromosome.
    Higher fitness is better (inverse of function value).
    """
    return 1 / (1 + f(chromosome.reshape(1, -1))[0])

def init_population(n_chromosomes, chromosome_length):
    """
    Initialize population with random values between -5 and 5.
    Args:
        n_chromosomes: Number of chromosomes in population
        chromosome_length: Length of each chromosome
    Returns:
        numpy array of shape (n_chromosomes, chromosome_length)
    """
    return np.random.uniform(-5, 5, (n_chromosomes, chromosome_length))

def select(population, fitness_func):
    """
    Select two parents using fitness proportionate selection.
    Args:
        population: numpy array of chromosomes
        fitness_func: function to calculate fitness
    Returns:
        tuple of two selected chromosomes
    """
    fitness_values = np.array([fitness_func(chrom) for chrom in population])
    probs = fitness_values / np.sum(fitness_values)
    selected_idx = np.random.choice(len(population), size=2, p=probs)
    return population[selected_idx[0]], population[selected_idx[1]]

def crossover(chromosome1, chromosome2):
    """
    Perform single-point crossover between two chromosomes.
    Args:
        chromosome1, chromosome2: parent chromosomes
    Returns:
        two child chromosomes
    """
    crossover_point = np.random.randint(1, len(chromosome1))
    child1 = np.concatenate([chromosome1[:crossover_point], 
                            chromosome2[crossover_point:]])
    child2 = np.concatenate([chromosome2[:crossover_point], 
                            chromosome1[crossover_point:]])
    return child1, child2

def mutate(chromosome, prob_mutation):
    """
    Perform mutation on a chromosome.
    Args:
        chromosome: numpy array to mutate
        prob_mutation: probability of mutation for each gene
    Returns:
        mutated chromosome
    """
    mutation_mask = np.random.random(len(chromosome)) < prob_mutation
    chromosome_copy = chromosome.copy()
    chromosome_copy[mutation_mask] = np.random.uniform(-5, 5, np.sum(mutation_mask))
    return chromosome_copy

def genetic_algorithm(population_size, chromosome_length=10, n_generations=300, prob_mutation=0.1):
    """Optimized genetic algorithm implementation using helper functions."""
    # Initialize population using helper function
    population = init_population(population_size, chromosome_length)
    
    for i in range(n_generations):
        # Compute fitness values for entire population at once
        fitness_values = np.array([fitness(chrom) for chrom in population])
        selection_probs = fitness_values / np.sum(fitness_values)
        
        # Create new population
        new_population = np.zeros_like(population)
        
        # Elitism: Keep the best solution
        best_idx = np.argmin(f(population))
        new_population[0] = population[best_idx]
        
        # Generate all parent indices at once for better performance
        parent_indices = np.random.choice(
            population_size, 
            size=(population_size-1, 2), 
            p=selection_probs
        )
        
        # Fill the rest of the population efficiently
        for j in range(1, population_size):
            # Get parents
            parent1 = population[parent_indices[j-1, 0]]
            parent2 = population[parent_indices[j-1, 1]]
            
            # Crossover and mutation
            child1, _ = crossover(parent1, parent2)
            new_population[j] = mutate(child1, prob_mutation)
        
        # Update population
        population = new_population
    
    # Return best solution
    return population[np.argmin(f(population))]

def population_size_experiment():
    """Experiment with different population sizes."""
    population_sizes = [100, 200, 300, 400, 500, 600,700, 800, 900, 1000]
    results = []
    
    for size in population_sizes:
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Run genetic algorithm
        solution = genetic_algorithm(population_size=size)
        
        # Store function value of best solution
        result_value = f(solution.reshape(1, -1))[0]
        results.append(result_value)
        print(f'Population size {size}: Function value = {result_value}')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(population_sizes, results, marker='o')
    plt.title('Function Value vs Population Size')
    plt.xlabel('Population Size')
    plt.ylabel('Function Value')
    plt.grid(True)
    plt.show()
    
    return results

# Run the experiment
if __name__ == "__main__":
    results = population_size_experiment()
    print('Function values for different population sizes:', results)