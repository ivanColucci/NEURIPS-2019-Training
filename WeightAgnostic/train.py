import os
import random
from NEAT.utils.filereporter import FileReporter
import neat
import pickle
from NEAT.my_reproduction import TournamentReproduction
import numpy as np
from NEAT.utils.utilities import eval_genome

# randomness
random.seed(1234)
np.random.seed(1234)

# constants
n_max_gen = 1000
n_workers = 100


def run(config_file, out_file='winner_genome', restore_checkpoint=False, checkpoint='neat-checkpoint'):
    config = neat.Config(neat.DefaultGenome, TournamentReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, "output.txt"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    pe = neat.ParallelEvaluator(n_workers, eval_genome)
    winner = p.run(pe.evaluate, n_max_gen)
    print(winner)
    # for key_id in winner.connections.keys():
    #     print(winner.connections[key_id].weight)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)


def start(out_file, restore_checkpoint=False, checkpoint='neat-checkpoint'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path, restore_checkpoint=restore_checkpoint, out_file=out_file, checkpoint=checkpoint)


if __name__ == '__main__':
    start('winner_genome',restore_checkpoint=False)
