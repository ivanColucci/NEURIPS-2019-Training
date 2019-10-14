import os
import random
from myenv import RewardShapingEnv
from NEAT.filereporter import FileReporter
import neat
import pickle
from NEAT.my_reproduction import TournamentReproduction
import numpy as np
from NEAT.parallel import ParallelEvaluator
from NEAT.my_genome import MyGenome
from NEAT.my_population import MyPopulation
# randomness
random.seed(1234)
np.random.seed(1234)

# Costants
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
    -1.745329251994329478e-01])
n_max_gen = 200
n_workers = 1


def add_action_for_3d(action):
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
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for_3d(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        final_rew += reward
        if done:
            break

    return [("fitness", final_rew), ("falcata", 3.2)]


def eval_genome(genome, config):
    env = RewardShapingEnv(visualize=False, seed=1234, difficulty=2)
    env.change_model(model='2D', difficulty=2, seed=1234)
    env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return execute_trial(env, net, 1000)


def run(config_file, rep_type=2, gen_type=2):

    # Load configuration.
    if rep_type == 2:
        rep_class = TournamentReproduction
    else:
        rep_class = neat.DefaultReproduction
    if gen_type == 2:
        gen_class = MyGenome
        pop_class = MyPopulation
    else:
        gen_class = neat.DefaultGenome
        pop_class = neat.Population


    config = neat.Config(gen_class, rep_class,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = pop_class(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-35')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, "output.txt"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    pe = ParallelEvaluator(n_workers, eval_genome)
    winner = p.run(pe.evaluate, n_max_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner
    with open('winner_genome_keep_alive', 'wb') as f:
        pickle.dump(winner, f)

def start():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-osim')
    run(config_path, rep_type=2, gen_type=2)

if __name__ == '__main__':
    start()
