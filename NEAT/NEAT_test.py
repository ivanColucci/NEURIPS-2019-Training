import os
import random
import neat
import pickle
from NEAT.utilities import eval_genome
random.seed(1234)


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-osim')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    if load_from_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        pe = neat.ParallelEvaluator(1, eval_genome)
        winner = p.run(pe.evaluate, 1)
        with open(source, 'wb') as f:
            pickle.dump(winner, f)
    else:
        with open(source, 'rb') as f:
            winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    result = eval_genome(winner_net, config, visual=True, is_a_net=True)
    print(result)


if __name__ == '__main__':
    test(source='winner_genome_distance_reward', load_from_checkpoint=False)
    # test(source='winner_genome_distance', load_from_checkpoint=True, checkpoint='neat-checkpoint-232')