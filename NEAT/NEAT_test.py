import math
import os
import numpy as np
import random
random.seed(1234)

import neat
from osim.env import L2M2019Env
import pickle

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
env = L2M2019Env(visualize=True, seed=1234, difficulty=2)
env.change_model(model='2D', difficulty=2, seed=None)
env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
env.spec.timestep_limit = timstep_limit


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
    t = 0
    for i in range(steps):
        t += sim_dt
        action = net.activate(observation)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        final_rew += reward
        if done:
            break
    return final_rew


def eval_genome(genome, config):
    # Returns the phenotype associated to given genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return execute_trial(env, net, 1000)


"""*************************************************MAIN*************************************************"""
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-osim')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

load_from_checkpoint = False

if load_from_checkpoint:
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-81')
    pe = neat.ParallelEvaluator(1, eval_genome)
    winner = p.run(pe.evaluate, 1)
    with open('winner_genome_CP', 'wb') as f:
        pickle.dump(winner, f)
else:
    with open('winner_genome', 'rb') as f:
        winner = pickle.load(f)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


final_rew = 0
observation = env.get_observation()
# Returns the phenotype associated to given genome
t = 0
for i in range(1000):
    t += sim_dt
    action = winner_net.activate(observation)
    action = add_action_for3D(action)
    obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
    final_rew += reward
    if done:
        break
print(final_rew)