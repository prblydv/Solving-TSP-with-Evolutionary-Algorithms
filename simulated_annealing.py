import random 
import math
from tsp_problem import TSPProblem

class SimulatedAnnealingTSP:

    """Simulated Annealing algorithm for solving TSP."""




    def __init__(self, tsp_problem, initial_temp=10000, cooling_rate=0.003, stopping_temp=0.1):


        self.tsp_problem = tsp_problem 
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp


    def run(self):
        """Runs the Simulated Annealing algorithm."""


        current_solution = random.sample(self.tsp_problem.cities, len(self.tsp_problem.cities))
        current_route = [city.id for city in current_solution]

        current_distance = self.tsp_problem.total_distance(current_route)

        best_solution, best_distance = current_route, current_distance

        temperature = self.initial_temp 

        while temperature > self.stopping_temp:
            new_route = current_route[:] 
            i, j = random.sample(range(len(new_route)), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_distance = self.tsp_problem.total_distance(new_route)


            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
                current_route = new_route
                current_distance = new_distance
                if current_distance < best_distance:
 
                    best_solution, best_distance = current_route, current_distance
 
            temperature *= (1 - self.cooling_rate)


        return best_solution, best_distance