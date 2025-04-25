# emissions_tracking.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample route data (from GA)
routes = {
    'Optimized': [0, 1, 2, 3, 0],  # Example route
    'Baseline': [0, 3, 1, 2, 0]   # Suboptimal route
}
distances = np.array([[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]])

# Emissions calculation (kg CO2/km)
EMISSIONS_PER_KM = 0.2  # Typical for delivery vans

def calculate_emissions(route, distances):
    total_distance = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    return total_distance * EMISSIONS_PER_KM

# Compute emissions
emissions = {name: calculate_emissions(route, distances) for name, route in routes.items()}

# Visualize
sns.barplot(x=list(emissions.values()), y=list(emissions.keys()))
plt.title('CO2 Emissions Comparison')
plt.xlabel('Emissions (kg CO2)')
plt.savefig('visualizations/emissions_comparison.png')
plt.close()

if __name__ == "__main__":
    for name, emission in emissions.items():
        print(f"{name} Route Emissions: {emission:.2f} kg CO2")