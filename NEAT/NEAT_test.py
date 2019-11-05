import os
import random
import neat
import pickle

from NEAT.utils.utilities import Evaluator
from NEAT.my_reproduction import TournamentReproduction
from NEAT.utils.my_checkpointer import MyCheckpointer
import numpy as np

random.seed(1234)
np.random.seed(1234)


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint', old_input=False):
    local_dir = os.path.dirname(__file__)
    if old_input:
        config_path = os.path.join(local_dir, 'Configs/config-osim')
    else:
        config_path = os.path.join(local_dir, '../WeightAgnostic/config')
    config = neat.Config(neat.DefaultGenome, TournamentReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    evaluator = Evaluator(reward_type=2, visual=True, is_a_net=True, old_input=False,
                          load_simulation=True, save_simulation=False, file_to_load="actions_step")
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
    evaluator = Evaluator(reward_type=5, visual=True, is_a_net=True, old_input=False,
                          load_simulation=True, save_simulation=False, file_to_load="actions_wnode03")
    result = evaluator.eval_genome(None, None)
    print("valore di fitness:", result)


if __name__ == '__main__':
    load_simulation()
    # test(source='../winner_genome', load_from_checkpoint=False, checkpoint='../neat-checkpoint-99')