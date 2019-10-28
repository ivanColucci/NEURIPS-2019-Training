import os
import random

from NEAT.utils.my_checkpointer import MyCheckpointer
from NEAT.utils.filereporter import FileReporter
import neat
import pickle
from NEAT.my_reproduction import TournamentReproduction
import numpy as np
from PSO.algorithms.my_local_best_PSO import MyLocalBestPSO
from PSO.algorithms.my_MOPSO import MOPSO
from PSO.algorithms.my_global_best_PSO import MyGlobalBestPSO
from NEAT.my_genome import MyGenome
from NEAT.my_population import MyPopulation
from NEAT.utils.utilities import Evaluator
random.seed(1234)
np.random.seed(1234)
evaluator = Evaluator(reward_type=1, old_input=False)

# constants
n_max_gen = 20
step_neat_gen = 50
step_pso_gen = 50
step_pso_pop = 15
n_workers = None
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-osim_config0')
bound_weight = 3


def from_weights_to_genome(weights):
    with open('last_winner', 'rb') as f:
        winner = pickle.load(f)
    return insert_weights(weights, winner)


def insert_weights(weights, genome):
    i = 0
    for key_id in genome.connections.keys():
        genome.connections[key_id].weight = weights[i]
        i = i + 1
    return genome


def pso_fitness(x):
    genome = from_weights_to_genome(x[0])

    config = neat.Config(neat.DefaultGenome, TournamentReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    return [-evaluator.eval_genome(genome, config)]


def fitness_manager(xs):
    dimension = len(xs)
    results = []
    for i in range(dimension):
        results.append(pso_fitness([xs[i]])[0])
    return results


def get_bounds(num_of_weights):
    bounds_up = []
    bounds_down = []
    for i in range(num_of_weights):
        bounds_up.append(bound_weight)
        bounds_down.append(-bound_weight)
    return (bounds_down, bounds_up)


def run(config_file, rep_type='Tournament', gen_type='Default', restore_checkpoint=False):

    # Load configuration.
    if rep_type == 'Tournament':
        rep_class = TournamentReproduction
    else:
        rep_class = neat.DefaultReproduction
    if gen_type == 'Default':
        gen_class = neat.DefaultGenome
        pop_class = neat.Population
    else:
        gen_class = MyGenome
        pop_class = MyPopulation

    config = neat.Config(gen_class, rep_class,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-35')
    else:
        p = pop_class(config)
    name_run = "output.txt"

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, name_run))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(MyCheckpointer(checkpoint_interval=50))
    pe = neat.ParallelEvaluator(n_workers, evaluator.eval_genome)

    winner = None
    for i in range(n_max_gen):
        winner = p.run(pe.evaluate, step_neat_gen)

        with open('last_winner', 'wb') as f:
            pickle.dump(winner, f)
        best_genome_weight = []
        genome_key_set = []
        for key_id in winner.connections.keys():
            genome_key_set.append(key_id)
            best_genome_weight.append(winner.connections[key_id].weight)

        fitness = fitness_manager
        dimension = len(best_genome_weight)
        bounds = get_bounds(dimension)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        optimizer = MyGlobalBestPSO(n_particles=step_pso_pop, dimensions=dimension, options=options, bounds=bounds)
        optimizer.set_reporter_name(name_run)
        optimizer.swarm.position[0] = best_genome_weight
        cost, pos = optimizer.optimize(fitness, iters=step_pso_gen)
        print(cost, p.best_genome.fitness)
        if -cost > p.best_genome.fitness:
            p.population[p.best_genome.key] = insert_weights(pos, p.population[p.best_genome.key])

    # Save the winner
    with open('winner_genome', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    run(config_path, rep_type='Tournament', gen_type='Default', restore_checkpoint=False)
