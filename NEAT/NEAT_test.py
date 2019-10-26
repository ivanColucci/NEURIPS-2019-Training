import os
import random
import neat
import pickle

from NEAT.utils.utilities import Evaluator
from NEAT.my_reproduction import TournamentReproduction
from NEAT.utils.my_checkpointer import MyCheckpointer
random.seed(1234)


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint', old_input=False):
    local_dir = os.path.dirname(__file__)
    if old_input:
        config_path = os.path.join(local_dir, 'Configs/config-osim')
    else:
        config_path = os.path.join(local_dir, '../WeightAgnostic/config')
    config = neat.Config(neat.DefaultGenome, TournamentReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    evaluator = Evaluator(reward_type=1, visual=True, is_a_net=True, old_input=False)
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
    print(result)

if __name__ == '__main__':
    test(source='../winner_checkpoint_99', load_from_checkpoint=True, checkpoint='../neat-checkpoint-99')
    # test(source='winner_genome_distance', load_from_checkpoint=True, checkpoint='neat-checkpoint-232')