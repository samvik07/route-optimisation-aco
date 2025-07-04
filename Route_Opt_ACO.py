# Import libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Define parameters for Ant Colony Optimisation (ACO) approach
NUM_ANTS = 20       # Number of ants (agents exploring routes)
NUM_ITERATIONS = 100  # Number of iterations for convergence
ALPHA = 1.0         # Influence of pheromone
BETA = 2.0          # Influence of heuristic (distance)
EVAPORATION = 0.5   # Pheromone evaporation rate
Q = 100             # Pheromone deposit factor

np.random.seed(42)


# Load data
def load_data(file_path):
    """
    This function reads a CSV file containing stop data with columns for stop ID, X (latitude), and Y (longitude),
    to load data into a dictionary format.

    Args:
    - file_path (str): Path to the CSV file containing the stop data.

    Returns:
    - dict: A dictionary where the key is the 'Stop-ID' and the value is a dictionary of 'X' and 'Y' coordinates.
    """

    # Open the file and read the first line to check if there is a header
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split(' ')

    # Skip the header row if present
    has_header = all(isinstance(val, str) for val in first_line)
    skip_rows = 1 if has_header else 0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, skiprows=skip_rows, names=['Stop-ID', 'X', 'Y'])

    # Convert the DataFrame to a dictionary, with 'Stop-ID' as the index and the corresponding 'X' and 'Y' values
    return df.set_index('Stop-ID').to_dict(orient='index')


# Compute Euclidean distance
def euclidean_distance(point1, point2):
    """
    This function calculate the Euclidean distance between two points in a 2D space.

    Args:
    - point1 (dict): A dictionary representing the first point, with keys 'X' and 'Y' for its coordinates.
    - point2 (dict): A dictionary representing the second point, with keys 'X' and 'Y' for its coordinates.

    Returns:
    - float: The Euclidean distance between the two points.
    """

    # Calculate the squared difference, sum them, and apply square root to get the Euclidean distance
    return np.sqrt((point1['X'] - point2['X'])**2 + (point1['Y'] - point2['Y'])**2)


# Create the distance matrix
def create_distance_matrix(stops):
    """
    This function creates a distance matrix for each pair of stops based on Euclidean distances.

    Args:
    - stops (dict): A dictionary where the keys are stop IDs and the values are dictionaries
                    containing 'X' and 'Y' coordinates of the stops.

    Returns:
    - dict: A dictionary representing the distance matrix, where each key is a tuple (stop1, stop2)
            and the value is the Euclidean distance between those two stops.
    """

    # Get list of stop IDs
    stop_ids = list(stops.keys())
    dist_matrix = {}

    # Loop through each pair of stops (i, j) to compute the distances
    for i in range(len(stop_ids)):
        for j in range(len(stop_ids)):
            if i != j:
                stop1 = stop_ids[i]
                stop2 = stop_ids[j]

                # Calculate the Euclidean distance between stop1 and stop2 and store it in the matrix
                dist_matrix[(stop1, stop2)] = euclidean_distance(stops[stop1], stops[stop2])

    return dist_matrix


# Ant Colony Optimisation (ACO) Algorithm for Route Optimisation:

# This implementation is inspired by concepts from the following sources, with modifications to align
# with the problem statement, such as the addition of refuelling stops after every 9 delivery stops:
#
# Source 1: "Ant Colony Optimization to Solve the Travelling Salesman Problem" by Dr. Tri Basuki Kurniawan,
# published in The Lorry Data, Tech & Product, dated 15th Feb 2022. Reference:
# (https://medium.com/thelorry-product-tech-data/ant-colony-optimization-to-solve-the-travelling-salesman-problem-d19bd866546e).
#
# Source 2: "Implementation of Ant Colony Optimization Using Python – Solve Traveling Salesman Problem"
# by Induraj S, dated 23rd Feb 2023. Reference:
# (https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475).


def ant_colony_optimisation(stops, depot_id=0):
    """
    This function finds the shortest delivery route using the Ant Colony Optimisation (ACO) algorithm.
    The algorithm uses pheromone trails to guide the search for optimal routes and incorporates both 
    exploration and exploitation strategies.

    Parameters:
    stops (dict): Dictionary containing the coordinates of stops (delivery and petrol stations).
    depot_id (int): ID of the starting point (depot), default is 0.

    Returns:
    tuple: The best route (list of stop IDs) and the total distance of that route (float).
    """

    # List of all stop IDs
    stop_ids = list(stops.keys())
    # num_stops = len(stop_ids)

    # Separate lists of delivery stops (1 to 48) and petrol stations (ID >= 101)
    delivery_stops = [sid for sid in stop_ids if 1 <= sid < 49]  # Delivery stops
    petrol_stations = [sid for sid in stop_ids if sid >= 101]  # Petrol stations

    # Create the distance matrix containing the distances between every pair of stops)
    dist_matrix = create_distance_matrix(stops)

    # Initialise pheromone values for all stop pairs
    pheromones = {(i, j): 1.0 for i in stop_ids for j in stop_ids if i != j}

    # Initialise best route to none and best distance to a very large number
    best_route = None
    best_distance = float('inf')

    # Main loop for each iteration
    for iteration in range(NUM_ITERATIONS):
        all_routes = []
        all_distances = []

        # Loop over each ant
        for ant in range(NUM_ANTS):

            # Start from depot and assign current position to depot
            route = [depot_id]
            unvisited = set(delivery_stops)
            current = depot_id

            # Track delivery stops, petrol stations, and last inserted stop
            delivery_count = 0
            inserted_petrol_stations = 0
            last_inserted_stop = depot_id

            # Ant's route generation loop
            while unvisited:
                # Probabilistically select next stop based on pheromone & heuristic (distance)
                probabilities = []
                total_pheromone = 0.0
                for stop in unvisited:
                    # Influence of phermone
                    tau = pheromones[(current, stop)] ** ALPHA
                    # Influence of heuristic
                    eta = (1 / dist_matrix[(current, stop)]) ** BETA
                    total_pheromone += tau * eta
                    probabilities.append((stop, tau * eta))

                # Normalise probabilities
                probabilities = [(stop, prob / total_pheromone) for stop, prob in probabilities]

                # Choose the next stop probabilistically (weighted by pheromone and distance)
                next_stop = random.choices([p[0] for p in probabilities], [p[1] for p in probabilities])[0]

                # Ensure that the next stop is not already in the route
                if next_stop not in route:
                    # Add the selected stop to the route and mark it as visited
                    route.append(next_stop)
                    unvisited.remove(next_stop)

                    # Update the current stop
                    current = next_stop

                    # Increment delivery count
                    delivery_count += 1

                # Insert petrol station after every 9 deliveries (i.e., at 10th, 20th, 30th, 40th, and 50th.)
                if delivery_count % 9 == 0 and inserted_petrol_stations < 5:

                    # Find the nearest petrol station to the current delivery stop
                    nearest_petrol = min(petrol_stations, key=lambda ps: dist_matrix[(current, ps)])

                    # Ensure no consecutive petrol station insertions
                    if last_inserted_stop != nearest_petrol:

                        # Add the nearest petrol station to the route
                        route.append(nearest_petrol)
                        last_inserted_stop = nearest_petrol
                        inserted_petrol_stations += 1

            # Return to the depot after completing all deliveries and petrol station insertions
            route.append(depot_id)

            # Calculate total route distance by summing up all the distances
            total_distance = sum(dist_matrix[(route[i], route[i + 1])] for i in range(len(route) - 1))

            # Store the route and corresponding route distance
            all_routes.append(route)
            all_distances.append(total_distance)

            # Update the best route if a shorter one is found
            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        # Pheromone update (evaporation and reinforcement)
        for i in stop_ids:
            for j in stop_ids:
                if i != j:
                    pheromones[(i, j)] *= (1 - EVAPORATION)  # Evaporation - Reduce pheromone level

        # Reinforce the pheromone levels for the routes taken by ants (positive reinforcement)
        for route, distance in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromones[(route[i], route[i + 1])] += Q / distance  # Add pheromone based on route quality

    # Return the best route found and its total distance
    return best_route, best_distance


# Plot Optimised Route
def plot_route(route, stops, route_distance):
    """
    This function plots the optimised delivery route as straight lines connecting the depot, delivery 
    destinations, and petrol stations, while displaying the total route distance.

    Parameters:
    route (list): List of stop IDs representing the optimised delivery route.
    stops (dict): Dictionary containing the coordinates (X, Y) of each stop. The keys are stop IDs.
    route_distance (float): The total distance of the optimised route.

    Returns:
    None: Displays a plot of the route.
    """

    plt.figure(figsize=(10, 8))

    # Plot different types of stops
    depot = stops[0]
    deliveries = [coords for sid, coords in stops.items() if 1 <= sid < 100]
    petrol_stations = [coords for sid, coords in stops.items() if sid >= 101]

    # Plot the depot as a red square ('s' marker)
    plt.scatter(depot['X'], depot['Y'], c='red', marker='s', label='Depot')

    # Plot delivery destinations as blue circles ('o' marker)
    plt.scatter([d['X'] for d in deliveries], [d['Y'] for d in deliveries], c='blue', marker='o', label='Delivery Stop')

    # Plot petrol stations as green triangles ('^' marker)
    plt.scatter([p['X'] for p in petrol_stations], [p['Y'] for p in petrol_stations], c='green', marker='^', label='Petrol Station')

    # Draw the route by connecting the stops with black lines
    for i in range(len(route) - 1):
        p1, p2 = stops[route[i]], stops[route[i+1]]
        plt.plot([p1['X'], p2['X']], [p1['Y'], p2['Y']], 'k-')

    # Annotate the route with the stop IDs at each stop point
    for sid in route:
        plt.text(stops[sid]['X'], stops[sid]['Y'], str(sid), fontsize=8, ha='right')

    plt.title(f'ACO-Optimised Delivery Route\nTotal Distance: {route_distance:.2f}')
    plt.legend()
    plt.savefig("Shortest_Route.jpg")
    plt.show()


# Main function
def main(file_path):
    """
    This is the main function to load data, find the optimised route and the route distance, and plot the route.

    Args:
    - file_path (str): The path to the CSV file containing the stop data. The file should contain stop IDs and their coordinates.

    Returns:
    - None: This function does not return any value. It prints the optimised route, its length, and the total route distance,
            and displays a plot of the route.
    """

    # Load stop data from the provided CSV file and convert it into a dictionary of stop information
    stops = load_data(file_path)

    # Find the optimised delivery route based on the stop data using ACO approach
    optimised_route, route_distance = ant_colony_optimisation(stops)
    print("Optimised Route:", optimised_route)
    print(f"Total Route Distance: {route_distance:.2f}")

    # Plot the optimised route with the stop data and total route distance
    plot_route(optimised_route, stops, route_distance)


# Call the main function with the provided CSV file path
main('Route_Dataset.csv')
