import os
import random
from NEAT.utils.filereporter import FileReporter
import neat
import pickle
from NEAT.DefaultTournament.my_reproduction import TournamentReproduction
import numpy as np
from NEAT.utils.utilities import Evaluator

# randomness
random.seed(1234)
np.random.seed(1234)

# constants
n_max_gen = 200
n_workers = 50


def run(config_file, out_file='winner_genome', rep_type='Tournament', restore_checkpoint=False, checkpoint='neat-checkpoint'):

    # Load configuration.
    if rep_type == 'Tournament':
        rep_class = TournamentReproduction
    else:
        rep_class = neat.DefaultReproduction
    gen_class = neat.DefaultGenome
    pop_class = neat.Population
    PEvaluator = neat.ParallelEvaluator

    config = neat.Config(gen_class, rep_class,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    evaluator = Evaluator(reward_type=6, visual=False, is_a_net=True, old_input=False)
    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = pop_class(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(FileReporter(True, "output.txt"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    pe = PEvaluator(n_workers, evaluator.eval_genome)
    winner = p.run(pe.evaluate, n_max_gen)

    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)

def start(out_file, restore_checkpoint=False, checkpoint='neat-checkpoint'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Configs/config-osim')
    run(config_path, rep_type='Tournament', restore_checkpoint=restore_checkpoint, out_file=out_file, checkpoint=checkpoint)

if __name__ == '__main__':
    start('winner_genome_16_10',restore_checkpoint=False)
