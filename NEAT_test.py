import os
import random
import neat
import pickle

from NEAT.utils.utilities import Evaluator
from NEAT.utils.my_checkpointer import MyCheckpointer
from NEAT.DefaultTournament.my_genome import MyGenome
from NEAT.EliteTournament.elite_reproduction import EliteReproduction
import numpy as np

random.seed(1234)
np.random.seed(1234)


def step_activation(z):
    if z > 0:
        return 1
    return 0


def reverse_activation(z):
    return -z


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint', old_input=False):
    local_dir = os.path.dirname(__file__)
    if old_input:
        config_path = os.path.join(local_dir, 'Configs/config-osim')
    else:
        config_path = os.path.join(local_dir, '../config_human0')
    config = neat.Config(MyGenome, EliteReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config.genome_config.add_activation('step', step_activation)
    config.genome_config.add_activation('reverse', reverse_activation)
    evaluator = Evaluator(reward_type=5, visual=True, is_a_net=True, old_input=False,
                          load_simulation=False, save_simulation=False, file_to_load="actions_6")
    if load_from_checkpoint:
        p = MyCheckpointer.restore_checkpoint(checkpoint)
        print(p.best_genome)
        for gid, g in p.population.items():
            if g.fitness is not None:
                winner = g
                break
    else:
        with open(source, 'rb') as f:
            winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    result = evaluator.eval_genome(winner_net, config)
    print("valore di fitness:", result)


def load_simulation():
    evaluator = Evaluator(reward_type=8, visual=True, is_a_net=True, old_input=False,
                          load_simulation=True, save_simulation=False,
                          file_to_load="../../actions_6.4", steps=1000)
    result = evaluator.eval_genome(None, None)
    print("valore di fitness:", result)


if __name__ == '__main__':
    load_simulation()
    # test(source='../winner_checkpoint_6', load_from_checkpoint=False, checkpoint='../neat-checkpoint-99')
