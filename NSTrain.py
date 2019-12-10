import os
import random
import neat
import pickle
import numpy as np
from NS.NSReproduction import NSReproduction
from NS.NSEvaluator import NSEvaluator
from NS.NSPopulation import NSPopulation
from NS.NSReporter import NSReporter
from NS.NSStagnation import NSStagnation
from NS.NSSpecies import NSSpeciesSet
from NS.NSGenome import NSGenome
from NS.NSElitePopulation import NSElitePopulation
from NS.NSEliteReproduction import NSEliteReproduction
from NS.NSCheckpointer import NSCheckpointer
from NEAT.utils.parallel_timeout import ParallelEvaluator


def run(config_file, out_file='winner_genome', n_workers=None, n_max_gen=None, checkpoint=None, winner='', seed=1234,
        elite=False, model='Walker2d-v3'):
    if elite:
        config = neat.Config(NSGenome, NSEliteReproduction,
                             NSSpeciesSet, NSStagnation,
                             config_file)
    else:
        config = neat.Config(NSGenome, NSReproduction,
                             NSSpeciesSet, NSStagnation,
                             config_file)

    train_num = ''
    evaluator = NSEvaluator(model_name=model, my_env=False, steps=5000, done=True, seed=seed)
    pe = ParallelEvaluator(n_workers, evaluator.eval_genome, timeout=500)
    if checkpoint is not None:
        p = NSCheckpointer.restore_checkpoint(checkpoint)
        for gid, g in p.population.items():
            g.fitness = None
    else:
        if elite:
            p = NSElitePopulation(config, n_neighbors=15, novelty_threshold=0.05, winner=winner)
        else:
            p = NSPopulation(config, n_neighbors=15, novelty_threshold=0.05, winner=winner)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(NSReporter(True, "output" + train_num + ".txt"))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(NSCheckpointer(checkpoint_interval=50, overwrite=True, filename_prefix='NS-checkpoint-' + train_num))
    winner = p.run(pe.evaluate, n_max_gen)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)


def start(out_file, restore_checkpoint=False, checkpoint='NS-checkpoint-', trials=1, elite=False):
    local_dir = os.path.dirname(__file__)
    if elite:
        config_path = os.path.join(local_dir, 'NSEliteHumanoidConfig')
    else:
        config_path = os.path.join(local_dir, 'NSHumanoidConfig0')
    for i in range(trials):
        seed = 1234 + i
        random.seed(seed)
        np.random.seed(seed)
        if restore_checkpoint:
            run(config_path, out_file=out_file, checkpoint=checkpoint, winner=str(i), elite=elite)
        else:
            run(config_path, out_file=out_file, winner=str(i), elite=elite)


if __name__ == '__main__':
    start('winner_genome', restore_checkpoint=False, trials=1, elite=True)
