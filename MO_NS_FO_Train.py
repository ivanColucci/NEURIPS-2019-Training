import os
import random
import neat
import pickle
import numpy as np
from MO_NS_FO.MO_NS_FO_Reproduction import Reproduction
from MO_NS_FO.MO_NS_FO_Evaluator import Evaluator
from MO_NS_FO.MO_NS_FO_Population import Population
from MO_NS_FO.MO_NS_FO_Reporter import Reporter
from MO_NS_FO.MO_NS_FO_Stagnation import Stagnation
from MO_NS_FO.MO_NS_FO_Species import SpeciesSet
from MO_NS_FO.MO_NS_FO_Genome import Genome
from MO_NS_FO.MO_NS_FO_ElitePopulation import ElitePopulation
from MO_NS_FO.MO_NS_FO_EliteReproduction import EliteReproduction
from MO_NS_FO.MO_NS_FO_Checkpointer import Checkpointer
from MO_NS_FO.MO_NS_FO_ParallelEvaluator import ParallelEvaluator


def run(config_file, out_file='winner_genome', n_workers=None, n_max_gen=None, checkpoint=None, winner='', seed=1234,
        elite=False):
    if elite:
        config = neat.Config(Genome, EliteReproduction,
                             SpeciesSet, Stagnation,
                             config_file)
    else:
        config = neat.Config(Genome, Reproduction,
                             SpeciesSet, Stagnation,
                             config_file)

    evaluator = Evaluator(my_env=False, steps=1000, done=True, seed=seed)
    pe = ParallelEvaluator(n_workers, evaluator.eval_genome, timeout=240)
    if checkpoint is not None:
        p = Checkpointer.restore_checkpoint(checkpoint)
        for gid, g in p.population.items():
            g.fitness = None
    else:
        if elite:
            # Per usare l'archivio decommentare la seguente istruzione e commentare quella successiva
            # p = ElitePopulation(config, n_neighbors=15, use_archive=True, winner=winner)
            p = ElitePopulation(config, n_neighbors=config.pop_size, winner=winner)
        else:
            p = Population(config, n_neighbors=config.pop_size, winner=winner)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(Reporter(True, "output" + winner + ".txt"))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(Checkpointer(checkpoint_interval=50, overwrite=True, filename_prefix='MO_NS_FO-checkpoint-' + winner))
    winner = p.run(pe.evaluate, n_max_gen)
    # Save the winner
    with open(out_file, 'wb') as f:
        pickle.dump(winner, f)


def start(out_file, restore_checkpoint=False, checkpoint='MO_NS_FO-checkpoint-', trials=1, elite=False):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'MO_NS_FO_EliteHumanoidConfig')
    for i in range(trials):
        seed = 1234 + i
        random.seed(seed)
        np.random.seed(seed)
        if restore_checkpoint:
            run(config_path, out_file=out_file, n_max_gen=500, checkpoint=checkpoint, winner=str(i), elite=elite)
        else:
            run(config_path, out_file=out_file, n_max_gen=500, winner=str(i), elite=elite)


if __name__ == '__main__':
    start('winner_genome', restore_checkpoint=False, trials=5, elite=True)
