from multiprocessing import Pool
from NEAT.utils.utilities import print_file
import numpy as np


def default_create_function(reproduction, key, genome_type, genome_config):
    g = genome_type(key)
    g.configure_new(genome_config)
    reproduction.ancestors[key] = tuple()
    return key, g

class ParallelCreator(object):
    def __init__(self, num_genomes, create_function=default_create_function, timeout=None, random_hidden=False):
        self.num_genomes = num_genomes
        self.create_function = create_function
        self.timeout = timeout
        self.pool = Pool(None)
        self.random_hidden = random_hidden

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def create_new(self, reproduction, genome_type, genome_config):
        jobs = []
        keys = []
        new_genomes = {}
        prev_hidden = genome_config.num_hidden

        for i in range(self.num_genomes):
            k = next(reproduction.genome_indexer)
            keys.append(k)

        for i in range(self.num_genomes):
            if self.random_hidden:
                n_hidden = np.random.randint(prev_hidden + 1, 2 * prev_hidden)
                genome_config.num_hidden = n_hidden
            jobs.append(self.pool.apply_async(self.create_function, (reproduction, keys[i], genome_type, genome_config)))

        # assign the fitness back to each genome
        for job in jobs:
            key, g = job.get(timeout=self.timeout)
            new_genomes[key] = g

        return new_genomes
