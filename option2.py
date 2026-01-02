import random

import math
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import wilcoxon


class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


# List of cities from the provided data
city_data = [
    (1, 6734, 1453),
    (2, 2233, 10),
    (3, 5530, 1424),
    (4, 401, 841),
    (5, 3082, 1644),
    (6, 7608, 4458),
    (7, 7573, 3716),
    (8, 7265, 1268),
    (9, 6898, 1885),
    (10, 1112, 2049),
    (11, 5468, 2606),
    (12, 5989, 2873),
    (13, 4706, 2674),
    (14, 4612, 2035),
    (15, 6347, 2683),
    (16, 6107, 669),
    (17, 7611, 5184),
    (18, 7462, 3590),
    (19, 7732, 4723),
    (20, 5900, 3561),
    (21, 4483, 3369),
    (22, 6101, 1110),
    (23, 5199, 2182),
    (24, 1633, 2809),
    (25, 4307, 2322),
    (26, 675, 1006),
    (27, 7555, 4819),
    (28, 7541, 3981),
    (29, 3177, 756),
    (30, 7352, 4506),
    (31, 7545, 2801),
    (32, 3245, 3305),
    (33, 6426, 3173),
    (34, 4608, 1198),
    (35, 23, 2216),
    (36, 7248, 3779),
    (37, 7762, 4595),
    (38, 7392, 2244),
    (39, 3484, 2829),
    (40, 6271, 2135),
    (41, 4985, 140),
    (42, 1916, 1569),
    (43, 7280, 4899),
    (44, 7509, 3239),
    (45, 10, 2676),
    (46, 6807, 2993),
    (47, 5185, 3258), 
    (48, 3023, 1942),
]

cities = [City(id, x, y) for id, x, y in city_data]


class TSPProblem:
    def __init__(self, cities):
        self.cities = cities

    def add_city(self, city):
        self.cities.append(city)

    def pseudo_euclidean_distance(self, city1, city2):
        dx = city1.x - city2.x
        dy = city1.y - city2.y
        return round(math.sqrt((dx ** 2 + dy ** 2) / 10.0))

    def total_distance(self, path):
        distance = 0
        for i in range(-1, len(path) - 1):

            city1 = self.cities[path[i] - 1]
            city2 = self.cities[path[i + 1] - 1]
            distance += self.pseudo_euclidean_distance(city1, city2)
        return distance

    def fitness(self, path):
        return 1 / float(self.total_distance(path))


class GeneticAlgorithmTSP:
    def __init__(self, tsp_problem, population_size=500, mutation_rate=0.001, crossover_rate=0.7, generations=2000, heuristic = True):
        self.tsp_problem = tsp_problem
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate  # New crossover rate parameter
        self.generations = generations
        self.heuristic = heuristic
        self.population = self.init_population()

    def init_population(self):
        population = []
        i = 0
        heuristic_sol = [48, 5, 29, 34, 41, 16, 22, 1, 8, 9, 38, 31, 44, 18, 46, 15, 12, 20, 33, 36, 7, 28, 6, 37, 19,
                         27, 30, 40, 2, 4, 26, 42, 39, 47, 43, 17, 21, 32, 24, 45, 35, 10, 14, 3, 23, 11, 13, 25]
        for _ in range(self.population_size):
            if i < 50 and self.heuristic is True:
                population.append(heuristic_sol)
            else:
                tour = list(range(1, len(self.tsp_problem.cities) + 1))
                random.shuffle(tour)
                population.append(tour)
            i += 1
        return population


    def mutate(self, genes):
        # print('gene',genes)
        for i in range(len(genes)):
            if random.random() < self.mutation_rate:
                a = random.randint(0, len(genes) - 1)
                b = random.randint(0, len(genes) - 1)
                genes[a], genes[b] = genes[b], genes[a]
        return genes

    def crossover(self, genes1, genes2):
        if random.random() < self.crossover_rate:
            start = random.randint(0, len(genes1) - 1)
            end = random.randint(start + 1, len(genes2))
            new_genes = genes1[start:end]
            for gene in genes2:
                if gene not in new_genes:
                    new_genes.append(gene)
            return new_genes
        else:
            return genes1 if random.random() < 0.5 else genes2

    def pick_selection(self, population, fitnesses, tournament_size=3):

        tournament_size = min(tournament_size, len(self.population))
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        best_index = max(tournament_indices, key=lambda idx: fitnesses[idx])
        return population[best_index]

    def natural_selection(self):
        next_population = []
        fitnesses = [self.tsp_problem.fitness(tour) for tour in self.population]
        for _ in range(self.population_size):
            parent1 = self.pick_selection(self.population, fitnesses)
            parent2 = self.pick_selection(self.population, fitnesses)
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            next_population.append(offspring)
        self.population = next_population
 
    def run(self):
        best_tour = None
        best_fitness = float('-inf')
        for generation in range(self.generations):
            self.natural_selection()
            current_best = max(self.population, key=lambda tour: self.tsp_problem.fitness(tour))
            current_best_fitness = self.tsp_problem.fitness(current_best)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_tour = current_best
            if generation % 50 == 0 or generation == self.generations - 1:
                print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Distance = {1 / best_fitness}")
        return best_tour


def simulated_annealing(tsp_problem, initial_temp=320000, cooling_rate=0.001, stopping_temp=0.1):
    current_solution = random.sample(tsp_problem.cities, len(tsp_problem.cities))
    current_route = [city.id for city in current_solution]
    current_distance = tsp_problem.total_distance(current_route)

    best_solution = list(current_route)
    best_distance = current_distance

    temperature = initial_temp
    runs = 0

    while temperature > stopping_temp or runs < 10000:
        new_route = list(current_route)

        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_distance = tsp_problem.total_distance(new_route)

        # deciding weather to accept the new solution
        if new_distance < current_distance or random.random() < math.exp(
                (current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance
            if current_distance < best_distance:
                best_solution = current_route
                best_distance = current_distance
        temperature *= (1 - cooling_rate)
        runs += 1

    return best_solution, best_distance


def plot_paths(cities, original_path, optimized_path):
    x_coords, y_coords = {}, {}
    for city in cities:
        x_coords[city.id] = city.x
        y_coords[city.id] = city.y

    orig_x = [x_coords[city_id] for city_id in original_path + [original_path[0]]]
    orig_y = [y_coords[city_id] for city_id in original_path + [original_path[0]]]
    opt_x = [x_coords[city_id] for city_id in optimized_path + [optimized_path[0]]]
    opt_y = [y_coords[city_id] for city_id in optimized_path + [optimized_path[0]]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7)) 

    axes[0].plot(orig_x, orig_y, 'o-', mfc='g', label='Simulated Annealing')
    axes[0].set_title('Optimized Path')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')

    # Plot optimized path
    axes[1].plot(opt_x, opt_y, 'o-', mfc='g', label='Genetic Algorithm')
    axes[1].set_title('Optimized Path')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')

    for ax in axes: 
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def tune_sa_parameters(tsp_problem):

    sa_param_options = {
        'initial_temp': [1000, 5000, 10000, 10000 * 2, 50000, 50000 * 2],
        'cooling_rate': [0.01, 0.003, 0.005, 0.0001],
        'stopping_temp': [0.01, 0.1, 1]
    }
    best_params = None
    best_distance = float('inf')

    results = []

    for init_temp in sa_param_options['initial_temp']:
        for cool_rate in sa_param_options['cooling_rate']:
            for stop_temp in sa_param_options['stopping_temp']:
                sa_params = {
                    'initial_temp': init_temp,
                    'cooling_rate': cool_rate,
                    'stopping_temp': stop_temp
                }
                _, distance = simulated_annealing(tsp_problem, init_temp, cool_rate, stop_temp)
                curr_res = [init_temp, cool_rate, stop_temp, distance]
                results.append(curr_res)
 
                if distance < best_distance:
                    best_distance = distance
                    best_params = sa_params

    return best_params, results


def tune_ga_parameters(tsp_problem):
    ga_param_options = {
        'population_size': [500],
        'mutation_rate': [0.001],
        'crossover_rate': [0.7], 
        'tournament_size': [500]
    }
    best_params = None 
    best_distance = float('inf')
 
    results = []

    for population_size in ga_param_options['population_size']:
        for mutation_rate in ga_param_options['mutation_rate']:
            for crossover_rate in ga_param_options['crossover_rate']:
                for tournament_size in ga_param_options['tournament_size']:
                    ga_params = {
                        'population_size': population_size,
                        'mutation_rate': mutation_rate,
                        'crossover_rate': crossover_rate,
                        'tournament_size': tournament_size
                    }
                    ga = GeneticAlgorithmTSP(tsp_problem, population_size, mutation_rate, crossover_rate,
                                             tournament_size)
                    best_tour = ga.run()
                    distance = tsp_problem.total_distance(best_tour)

                    curr_res = [population_size, mutation_rate, crossover_rate, tournament_size, distance]
                    results.append(curr_res)

                    if distance < best_distance:
                        best_distance = distance
                        best_params = ga_params

    return best_params, results


def plot_results(results):
    results_array = np.array(results)

    init_temps = results_array[:, 0]
    cool_rates = results_array[:, 1]

    stop_temps = results_array[:, 2]
    distances = results_array[:, 3]
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    scatter = ax[0].scatter(init_temps, distances, c=cool_rates, cmap='viridis')
    ax[0].set_xlabel('Initial Temperature') 
    ax[0].set_ylabel('Distance')
    ax[0].set_title('Initial Temperature vs Distance')
    legend1 = ax[0].legend(*scatter.legend_elements(), title="Cooling Rates")
    ax[0].add_artist(legend1)
    scatter = ax[1].scatter(cool_rates, distances, c=init_temps, cmap='coolwarm')
    ax[1].set_xlabel('Cooling Rate')
    ax[1].set_ylabel('Distance')
    ax[1].set_title('Cooling Rate vs Distance')
    legend2 = ax[1].legend(*scatter.legend_elements(), title="Initial Temps")
    ax[1].add_artist(legend2) 
    scatter = ax[2].scatter(stop_temps, distances, c=cool_rates, cmap='cividis')
    ax[2].set_xlabel('Stopping Temperature')
    ax[2].set_ylabel('Distance')
    ax[2].set_title('Stopping Temperature vs Distance')
    legend3 = ax[2].legend(*scatter.legend_elements(), title="Cooling Rates")
    ax[2].add_artist(legend3)
    plt.tight_layout()
    plt.show()


def plot_ga_results(results):
    results_array = np.array(results)

    population_sizes = results_array[:, 0]
    mutation_rates = results_array[:, 1]
    crossover_rates = results_array[:, 2]
    tournament_sizes = results_array[:, 3]
    distances = results_array[:, 4]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    scatter1 = axs[0, 0].scatter(population_sizes, distances, c=mutation_rates, cmap='viridis', label='Mutation Rates')
    axs[0, 0].set_xlabel('Population Size')
    axs[0, 0].set_ylabel('Distance')
    axs[0, 0].set_title('Population Size vs Distance')
    fig.colorbar(scatter1, ax=axs[0, 0], label='Mutation Rate')

    scatter2 = axs[0, 1].scatter(mutation_rates, distances, c=population_sizes, cmap='coolwarm',
                                 label='Population Sizes')
    axs[0, 1].set_xlabel('Mutation Rate')
    axs[0, 1].set_ylabel('Distance')
    axs[0, 1].set_title('Mutation Rate vs Distance')
    fig.colorbar(scatter2, ax=axs[0, 1], label='Population Size')

    scatter3 = axs[1, 0].scatter(crossover_rates, distances, c=tournament_sizes, cmap='cividis',
                                 label='Tournament Sizes')
    axs[1, 0].set_xlabel('Crossover Rate')
    axs[1, 0].set_ylabel('Distance')
    axs[1, 0].set_title('Crossover Rate vs Distance')
    fig.colorbar(scatter3, ax=axs[1, 0], label='Tournament Size')

    scatter4 = axs[1, 1].scatter(tournament_sizes, distances, c=crossover_rates, cmap='magma', label='Crossover Rates')
    axs[1, 1].set_xlabel('Tournament Size')
    axs[1, 1].set_ylabel('Distance')
    axs[1, 1].set_title('Tournament Size vs Distance')
    fig.colorbar(scatter4, ax=axs[1, 1], label='Crossover Rate')

    plt.tight_layout()
    plt.show()

 
def run_ga_multiple_times(tsp_problem, ga_params, num_runs=30, total_evaluations=10000):
    generations_needed = total_evaluations // ga_params['population_size']
    ga_params['generations'] = generations_needed

    distances = []
    for _ in range(num_runs):
        ga = GeneticAlgorithmTSP(tsp_problem, ga_params['population_size'], ga_params['mutation_rate'],
                                 ga_params['crossover_rate'], ga_params['generations'])

        best_tour = ga.run()
        distance = tsp_problem.total_distance(best_tour)
        distances.append(distance)


    average_distance = sum(distances) / num_runs
    std_deviation = np.std(distances)
    print(len(distances))
    return average_distance, std_deviation, distances


def run_sa_multiple_times(tsp_problem, sa_params, num_runs=30, max_iterations=10000):
    sa_params['max_iterations'] = max_iterations
    distances = []
    for _ in range(num_runs):
        _, distance = simulated_annealing(tsp_problem, sa_params['initial_temp'], sa_params['cooling_rate'],
                                          sa_params['stopping_temp'])
        distances.append(distance)

    average_distance = sum(distances) / num_runs
    std_deviation = np.std(distances)
    print(len(distances))
    return average_distance, std_deviation, distances


def plot_sa_distances(distances):
    runs = list(range(1, len(distances) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(runs, distances, marker='o', linestyle='-', color='blue')

    plt.title('Distances for 30 Runs of Simulated Annealing')
    plt.xlabel('Run')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.xticks(runs)
    plt.tight_layout()
    plt.show()


def plot_ga_distances(distances):
    runs = list(range(1, len(distances) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(runs, distances, marker='o', linestyle='-', color='blue')

    plt.title('Distances for 30 Runs of Genetic Algorithm')
    plt.xlabel('Run')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.xticks(runs) 
    plt.tight_layout()
    plt.show() 


if __name__ == "__main__":
    tsp_problem = TSPProblem(cities)
    time1 = time.time()
    sample_path = list(range(1, len(cities) + 1))

    print('Currently tuning GA parameters')
    ga_params, results_ga = tune_ga_parameters(tsp_problem)

    print('Currently tuning SA parameters')
    sa_params, results_sa = tune_sa_parameters(tsp_problem)


    print('Plotting Results')
    plot_results(results_sa)
    plot_results(results_ga)

    print('ga best params:', ga_params)
    print('sa best params: ', sa_params)
 
    time2 = time.time()
    time_taken = time1 - time2
    print('total time taken', time_taken)
    print('tuning parameters is completed, not staring the independent runs \n \n \n')

    ga_average_distance, ga_std_deviation, distances_ga = run_ga_multiple_times(tsp_problem, ga_params)
    sa_average_distance, sa_std_deviation, distances_sa = run_sa_multiple_times(tsp_problem, sa_params)

    print("GA - Average Distance:", ga_average_distance, "Standard Deviation:", ga_std_deviation)
    print("SA - Average Distance:", sa_average_distance, "Standard Deviation:", sa_std_deviation)
    print('Now Plotting the 30 distances')
    print('-------------------------------')
    print('GA - Distances:', distances_ga)
    print('SA - Distances:', distances_sa)

    plot_ga_distances(distances_ga) 
    plot_sa_distances(distances_sa)

    stat, p = wilcoxon(distances_ga, distances_sa)

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')