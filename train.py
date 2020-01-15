import os
import random
from NEAT.utils.filereporter import FileReporter
import neat
import pickle

import numpy as np
from NEAT.utils.utilities import Evaluator
from NEAT.utils.my_checkpointer import MyCheckpointer
from NEAT.utils.parallel_timeout import ParallelEvaluator
from NEAT.DefaultTournament.my_genome import MyGenome
from NEAT.EliteTournament.elite_population import ElitePopulation
from NEAT.EliteTournament.elite_reproduction import EliteReproduction

# randomness
random.seed(1234)
np.random.seed(1234)

# constants
n_max_gen = 500
n_workers = None


def step_activation(z):
    if z > 0:
        return 1
    return 0


def reverse_activation(z):
    return -z


def run(config_file, out_file='winner_genome', restore_checkpoint=False, checkpoint='neat-checkpoint', winner='0'):
    config = neat.Config(MyGenome, EliteReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config.genome_config.add_activation('step', step_activation)
    config.genome_config.add_activation('reverse', reverse_activation)

    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint:
        p = MyCheckpointer.restore_checkpoint(checkpoint, ElitePopulation)
    else:
        p = ElitePopulation(config, overwrite=True)
        p.allow_regeneration(False)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, "output" + winner + ".txt"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(MyCheckpointer(checkpoint_interval=50, filename_prefix='neat-checkpoint-' + winner, overwrite=True))
    #   1 - distance metric
    #   2 - area metric
    #   3 - step reward with a bonus for staying with the pelvis between 0.84 and 0.94
    #   4 - step reward
    #   5 - Definitive
    #   6 - body in range incremental
    evaluator = Evaluator(reward_type=5, visual=False, old_input=False, steps=1000)
    pe = ParallelEvaluator(n_workers, evaluator.eval_genome, timeout=240)
    winner = p.run(pe.evaluate, n_max_gen)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)


def start(out_file, restore_checkpoint=False, checkpoint='neat-checkpoint', trials=1):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_human0')
    for i in range(trials):
        seed = 1234 + i
        random.seed(seed)
        np.random.seed(seed)
        if restore_checkpoint:
            run(config_path, out_file=out_file, checkpoint=checkpoint, winner=str(i))
        else:
            run(config_path, out_file=out_file, winner=str(i))

if __name__ == '__main__':
    start('winner_genome', restore_checkpoint=False, checkpoint='neat-checkpoint-', trials=5)
