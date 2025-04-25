# route_optimization.py
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Parameters
NUM_LOCATIONS = 10
POP_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1

# Generate delivery locations
np.random.seed(42)
locations = np.vstack([np.array([50, 50]), np.random.rand(NUM_LOCATIONS, 2) * 100])

# Distance matrix
def compute_distances(locations):
    n = len(locations)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))
    return distances

distances = compute_distances(locations)

# Genetic Algorithm
def genetic_algorithm():
    # Initialize population with random routes
    population = [random.sample(range(1, NUM_LOCATIONS + 1), NUM_LOCATIONS) for _ in range(POP_SIZE)]
    best_route, best_distance = None, float('inf')

    for _ in range(GENERATIONS):
        # Calculate fitness for each route
        fitnesses = []
        for route in population:
            total_distance = distances[0, route[0]]  # Distance from depot to first location
            for i in range(len(route) - 1):
                total_distance += distances[route[i], route[i + 1]]
            total_distance += distances[route[-1], 0]  # Distance back to depot
            fitnesses.append(total_distance)

        # Update best route and distance
        min_distance = min(fitnesses)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = population[fitnesses.index(min_distance)]

        # Selection: Select parents based on fitness (lower distance is better)
        selected_population = random.choices(
            population, weights=[1 / f for f in fitnesses], k=POP_SIZE
        )

        # Crossover: Create new population
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = random.sample(selected_population, 2)
            cut = random.randint(1, NUM_LOCATIONS - 1)
            child1 = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
            child2 = parent2[:cut] + [gene for gene in parent1 if gene not in parent2[:cut]]
            new_population.extend([child1, child2])

        # Mutation: Randomly swap two locations in a route
        for route in new_population:
            if random.random() < MUTATION_RATE:
                i, j = random.sample(range(NUM_LOCATIONS), 2)
                route[i], route[j] = route[j], route[i]

        population = new_population

    # Return the best route with depot added at start and end
    return [0] + best_route + [0], best_distance

# Plot the optimized route
def plot_route(route, distance):
    os.makedirs('visualizations', exist_ok=True)  # Ensure directory exists
    plt.scatter(locations[1:, 0], locations[1:, 1], c='blue', label='Delivery Points')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', label='Depot', s=100)
    for i in range(len(route) - 1):
        start, end = locations[route[i]], locations[route[i + 1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-')
    plt.title(f'Optimized Route (Distance: {distance:.2f} km)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.savefig('visualizations/optimized_route.png')
    plt.close()

if __name__ == "__main__":
    route, distance = genetic_algorithm()
    plot_route(route, distance)
    print(f"Optimized Route: {route}, Distance: {distance:.2f} km")