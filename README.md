##   🚙  Vehicle Route Optimisation using Ant Colony Optimisation (ACO) 🐜

Python implementation of an Ant Colony Optimisation algorithm to solve the vehicle routing problem with periodic refueling constraints.

This project implements an Ant Colony Optimisation (ACO) algorithm in Python to solve a delivery route optimisation problem. It considers realistic constraints such as inserting petrol station visits after every 9 delivery stops and computes the most efficient route based on geographical coordinates.


###  📌 Features

- Optimises delivery routes using Ant Colony Optimisation (ACO)
- Inserts petrol station stops after every 9 deliveries
- Calculates and visualises the optimised route with matplotlib
- Outputs total route distance and saves the route plot


###  🧠 About the Algorithm (Ant Colony Optimization)

ACO is a bio-inspired optimisation technique modelled after the behaviour of real ants. In nature, ants find the shortest paths between their colony and food sources by laying down and following **pheromone trails**. Over time, shorter paths accumulate more pheromones, making them more attractive to other ants.

In this scenario, each ant simulates one possible route (solution) in each iteration. Pheromone levels and distance heuristics guide the selection of the next stop. Multiple ants explore different paths in parallel, and over time, the algorithm converges on the most efficient delivery route for the delivery vehicle.


#### ⚠️ Dataset Disclaimer

The original dataset used in this project cannot be shared publicly due to confidentiality.
However, the script expects a CSV file with the following structure:

| Stop-ID | X (Latitude) | Y (Longitude) | Comments                               |
|---------|--------------|---------------|----------------------------------------|
| 0       | 11.0         |  7.0          | Stop-ID 0 is Depot (Start and End Point|
| 8       | 10.0         | 12.7          | Stop-IDs 1-48 are Delivery Stops       |
| 101     | 15.2         | 13.1          | Stop-IDs above 100 are Petrol Stations |



#### 📖 References
This implementation is inspired by:
- "Ant Colony Optimization to Solve the Travelling Salesman Problem" by Dr. Tri Basuki Kurniawan,  published in The Lorry Data, Tech & Product, dated 15th Feb 2022.     (https://medium.com/thelorry-product-tech-data/ant-colony-optimization-to-solve-the-travelling-salesman-problem-d19bd866546e). 
- "Implementation of Ant Colony Optimization Using Python – Solve Traveling Salesman Problem" by Induraj S, dated 23rd Feb 2023.
  (https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475)
  
