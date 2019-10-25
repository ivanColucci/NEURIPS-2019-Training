import os
import random
import neat
import pickle
from NEAT.utils.utilities import eval_genome
from NEAT.my_reproduction import TournamentReproduction
random.seed(1234)


def test(source='winner_genome', load_from_checkpoint=False, checkpoint='neat-checkpoint', old_input=False):
    local_dir = os.path.dirname(__file__)
    if old_input:
        config_path = os.path.join(local_dir, 'Configs/config-osim')
    else:
        config_path = os.path.join(local_dir, '../WeightAgnostic/config')
    config = neat.Config(neat.DefaultGenome, TournamentReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    if load_from_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        pe = neat.ParallelEvaluator(None, eval_genome)
        winner = p.run(pe.evaluate, 1)
        with open(source, 'wb') as f:
            pickle.dump(winner, f)
    else:
        with open(source, 'rb') as f:
            winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    result = eval_genome(winner_net, config, visual=True, is_a_net=True, old_input=old_input)
    print(result)

if __name__ == '__main__':
    test(source='../winner_checkpoint_70', load_from_checkpoint=False, old_input=False)
    # test(source='winner_genome_distance', load_from_checkpoint=True, checkpoint='neat-checkpoint-232')