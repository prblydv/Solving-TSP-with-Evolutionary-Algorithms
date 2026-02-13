import math
from city import City

class TSPProblem: 
 
    """Encapsulates the TSP problem, including cities and distance calculations.""" 

    def __init__(self, cities):
        self.cities = cities 

    def pseudo_euclidean_distance(self, city1, city2):
        """Computes the pseudo-Euclidean distance between two cities.""" 

        dx = city1.x - city2.x

        dy = city1.y - city2.y 
        return round(math.sqrt((dx ** 2 + dy ** 2) / 10.0))  

 
    def total_distance(self, path):
        """Calculates the total distance for a given path."""  
        distance = 0

        for i in range(-1, len(path) - 1):
            city1 = self.cities[path[i] - 1] 



            city2 = self.cities[path[i + 1] - 1]
            distance += self.pseudo_euclidean_distance(city1, city2)
        return distance 

    def fitness(self, path): 
        """Computes fitness score (inverse of distance)."""
        return 1 / float(self.total_distance(path))  