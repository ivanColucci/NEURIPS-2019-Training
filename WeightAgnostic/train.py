import os
import random
from NEAT.utils.filereporter import FileReporter
import neat
import pickle
from NEAT.my_reproduction import TournamentReproduction
import numpy as np
from NEAT.utils.utilities import Evaluator
from WeightAgnostic.time_population import TimePopulation
from NEAT.utils.my_checkpointer import MyCheckpointer
from WeightAgnostic.parallel_timeout import ParallelEvaluator

# randomness
random.seed(1234)
np.random.seed(1234)

# constants
n_max_gen = 1000
n_workers = None


def run(config_file, out_file='winner_genome', restore_checkpoint=False, checkpoint='neat-checkpoint'):
    config = neat.Config(neat.DefaultGenome, TournamentReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = TimePopulation(config)
        p.allow_rigeneration(False)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, "output.txt"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(MyCheckpointer(checkpoint_interval=50))
    #   1 - distance metric
    #   2 - area metric
    #   3 - step reward with a bonus for staying with the pelvis between 0.84 and 0.94
    #   4 - step reward
    evaluator = Evaluator(reward_type=None, old_input=False, steps=1000)
    pe = ParallelEvaluator(n_workers, evaluator.eval_genome, timeout=500)
    winner = p.run(pe.evaluate, n_max_gen)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)

def start(out_file, restore_checkpoint=False, checkpoint='neat-checkpoint'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path, restore_checkpoint=restore_checkpoint, out_file=out_file, checkpoint=checkpoint)

if __name__ == '__main__':
    start('winner_genome', restore_checkpoint=False, checkpoint='neat-checkpoint-999')
