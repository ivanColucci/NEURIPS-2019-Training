import os
import random
import neat
import pickle
from NEAT.utils.utilities import Evaluator
from NEAT.my_reproduction import TournamentReproduction
random.seed(1234)


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '../WeightAgnostic/config')
    config = neat.Config(neat.DefaultGenome, TournamentReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    evaluator = Evaluator(reward_type=3, visual=True, is_a_net=True, old_input=False)
    if load_from_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        pe = neat.ParallelEvaluator(None, evaluator.eval_genome)
        winner = p.run(pe.evaluate, 1)
        with open(source, 'wb') as f:
            pickle.dump(winner, f)
    else:
        with open(source, 'rb') as f:
            winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    result = evaluator.eval_genome(winner_net, config)
    print(result)

if __name__ == '__main__':
    test(source='../winner_1', load_from_checkpoint=False)
    # test(source='winner_genome_distance', load_from_checkpoint=True, checkpoint='neat-checkpoint-232')