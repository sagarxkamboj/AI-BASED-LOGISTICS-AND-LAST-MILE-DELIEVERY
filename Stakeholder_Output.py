# stakeholder_output.py
import numpy as np

def calculate_emissions(route, distances, emission_factor=0.21):
    """
    Calculate estimated emissions for a given route.
    :param route: List of route indices.
    :param distances: 2D array of distances between locations.
    :param emission_factor: Emissions per km (default: 0.21 kg CO2/km).
    :return: Total emissions in kg CO2.
    """
    total_distance = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    return total_distance * emission_factor

def generate_route_summary(route, distances, emissions, filename='route_summary.txt'):
    """
    Generate a route summary and save it to a file.
    :param route: List of route indices.
    :param distances: 2D array of distances between locations.
    :param emissions: Total emissions in kg CO2.
    :param filename: Name of the file to save the summary.
    """
    total_distance = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    with open(filename, 'w') as f:
        f.write("Driver Route Summary\n")
        f.write(f"Route: {route}\n")
        f.write(f"Total Distance: {total_distance:.2f} km\n")
        f.write(f"Estimated Emissions: {emissions:.2f} kg CO2\n")
        f.write("Instructions: Start at depot (0), follow route, return to depot.\n")
    print(f"Route summary saved to {filename}")

if __name__ == "__main__":
    sample_route = [0, 1, 2, 3, 0]
    sample_distances = np.array([[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]])
    sample_emissions = calculate_emissions(sample_route, sample_distances)
    generate_route_summary(sample_route, sample_distances, sample_emissions)