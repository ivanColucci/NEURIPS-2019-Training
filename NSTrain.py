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
from NS.NSParallelEvaluator import ParallelEvaluator


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
    evaluator = NSEvaluator(model_name=model, my_env=False, steps=1000, done=True, seed=seed)
    pe = ParallelEvaluator(n_workers, evaluator.eval_genome, timeout=240)
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
    p.add_reporter(NSReporter(True, "NS_output" + winner + ".txt"))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(NSCheckpointer(checkpoint_interval=5, overwrite=True, filename_prefix='NS-checkpoint-' + winner))
    winner = p.run(pe.evaluate, n_max_gen)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)


def start(out_file, restore_checkpoint=False, checkpoint='NS-checkpoint-', trials=1, elite=False, offset=0):
    local_dir = os.path.dirname(__file__)
    if elite:
        config_path = os.path.join(local_dir, 'NSEliteHumanoidConfig')
    else:
        config_path = os.path.join(local_dir, 'NSHumanoidConfig0')
    for i in range(trials):
        seed = 1234 + i + offset
        random.seed(seed)
        np.random.seed(seed)
        if restore_checkpoint:
            run(config_path, out_file=out_file, n_max_gen=500, checkpoint=checkpoint, winner=str(i+offset), elite=elite)
        else:
            run(config_path, out_file=out_file, n_max_gen=500, winner=str(i+offset), elite=elite)


if __name__ == '__main__':
    start('winner_genome', restore_checkpoint=False, trials=10, elite=True, offset=0)
