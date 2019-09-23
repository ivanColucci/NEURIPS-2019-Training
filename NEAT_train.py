import os
import random
random.seed(1234)
from osim.env import L2M2019Env
import neat
import pickle
from parallel import ParallelEvaluator
import numpy as np


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
env = L2M2019Env(visualize=False, seed=None, difficulty=2)
env.change_model(model='2D', difficulty=2, seed=None)
env.reset(project=True, seed=None, obs_as_dict=False, init_pose=INIT_POSE)
env.spec.timestep_limit = timstep_limit
n_max_gen = 100
n_workers = 32


def execute_trial(env, net, steps):
    final_rew = 0
    observation = env.get_observation()
    # Returns the phenotype associated to given genome
    for i in range(steps):
        action = net.activate(observation)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        final_rew += reward
        if done:
            break
    return final_rew


def eval_genome(genome, config):
    # Returns the phenotype associated to given genome
    env.reset(project=True, seed=None, obs_as_dict=False, init_pose=INIT_POSE)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return execute_trial(env, net, 1000)


def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-198')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    pe = neat.ParallelEvaluator(n_workers, eval_genome)
    winner = p.run(pe.evaluate, n_max_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner
    with open('winner_genome', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-osim')
    run(config_path)
