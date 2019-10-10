import os
import random
from myenv import RewardShapingEnv
from NEAT.filereporter import FileReporter
import neat
import pickle
from NEAT.parallel import ParallelEvaluator
from NEAT.my_reproduction import TournamentReproduction
import numpy as np
from PSO.my_local_best_PSO import MyLocalBestPSO
import math
import copy
random.seed(1234)
np.random.seed(1234)

INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01]) # ankle flex

sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))
# Create the environment
# env = RewardShapingEnv(visualize=False, seed=1234, difficulty=2)
# env.set_reward_function(env.distance_reward)
# env.change_model(model='2D', difficulty=2, seed=1234)
# env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
# env.spec.timestep_limit = timstep_limit

n_max_gen = 30
step_neat_gen = 5
step_pso_gen = 30
step_pso_pop = 15
n_workers = 30


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
                         'config-osim_config0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = RewardShapingEnv(visualize=False, seed=1234, difficulty=2)
    env.set_reward_function(env.distance_reward)
    env.change_model(model='2D', difficulty=2, seed=1234)
    env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
    return [-execute_trial(env, net, 1000)]


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
        bounds_up.append(30)
        bounds_down.append(-30)
    return (bounds_down, bounds_up)


def add_action_for3D(action):
    Fmax_ABD = 4460.290481
    Fmax_ADD = 3931.8
    r_leg, l_leg = action[:9], action[9:]
    full_action = []
    full_action.append(0.1)
    full_action.append(0.1*Fmax_ADD/Fmax_ABD)
    for el in r_leg:
        full_action.append(el)
    full_action.append(0.1)
    full_action.append(0.1 * Fmax_ADD / Fmax_ABD)
    for el in l_leg:
        full_action.append(el)
    return full_action


def execute_trial(env, net, steps):
    final_rew = 0
    observation = env.get_observation()
    # Returns the phenotype associated to given genome
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for3D(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        final_rew += reward
        if done:
            break
    return final_rew


def eval_genome(genome, config):
    # for key_id in genome.connections.keys():
    #     print(genome.connections[key_id].weight)
    # Returns the phenotype associated to given genome
    env = RewardShapingEnv(visualize=False, seed=1234, difficulty=2)
    env.set_reward_function(env.distance_reward)
    env.change_model(model='2D', difficulty=2, seed=1234)
    env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return execute_trial(env, net, 1000)


def run(config_file, rep_type=2):

    # Load configuration.
    if rep_type == 2:
        rep_class = TournamentReproduction
    else:
        rep_class = neat.DefaultReproduction

    config = neat.Config(neat.DefaultGenome, rep_class,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-107')

    name_run = "output_10_10_config0.txt"

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, name_run))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    pe = neat.ParallelEvaluator(n_workers, eval_genome)


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
        optimizer = MyLocalBestPSO(n_particles=step_pso_pop, dimensions=dimension, options=options, bounds=bounds)
        optimizer.set_reporter_name(name_run)
        optimizer.swarm.position[0] = best_genome_weight
        cost, pos = optimizer.optimize(fitness, iters=step_pso_gen, n_processes=10)
        print(cost, p.best_genome.fitness)
        if -cost > p.best_genome.fitness:
            p.population[p.best_genome.key] = insert_weights(pos, p.population[p.best_genome.key])


    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # Save the winner
    with open('temp', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-osim')
    run(config_path, rep_type=2)
