import random
from tsp_problem import TSPProblem

class GeneticAlgorithmTSP:
    """Genetic Algorithm for solving TSP."""

    def __init__(self, tsp_problem, population_size=500, mutation_rate=0.001, crossover_rate=0.7, generations=2000):
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = self.init_population()

    def init_population(self):
        """Initializes a population of possible routes."""
        population = []
        for _ in range(self.population_size):
            tour = list(range(1, len(self.tsp_problem.cities) + 1))
            random.shuffle(tour) 
            population.append(tour)
        return population

    def mutate(self, genes):
        """Mutates a given chromosome."""
        for i in range(len(genes)):
            if random.random() < self.mutation_rate:
                a, b = random.sample(range(len(genes)), 2)
                genes[a], genes[b] = genes[b], genes[a]
        return genes 

    def crossover(self, parent1, parent2):

        """Performs crossover between two parents.""" 
        if random.random() < self.crossover_rate:
            start, end = sorted(random.sample(range(len(parent1)), 2))
            child = parent1[start:end]
            for gene in parent2:
                if gene not in child:
                    child.append(gene)
            return child
        return parent1 if random.random() < 0.5 else parent2

    def natural_selection(self):
        """Performs selection, crossover, and mutation to evolve the population."""
        fitnesses = [self.tsp_problem.fitness(tour) for tour in self.population]
        next_population = []
        for _ in range(self.population_size): 

            parent1 = random.choices(self.population, weights=fitnesses, k=1)[0]
            parent2 = random.choices(self.population, weights=fitnesses, k=1)[0]
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)

            next_population.append(offspring)

        self.population = next_population
  
    def run(self):
        """Runs the genetic algorithm."""
        best_tour = min(self.population, key=lambda tour: self.tsp_problem.total_distance(tour))
        return best_tour