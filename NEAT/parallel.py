from multiprocessing import Pool
import numpy as np

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, threshold_fitness, threshold_stagnation,  targets, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.targets = targets
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.current_best_fitness = 0
        self.threshold_fitness = threshold_fitness
        self.threshold_stagnation = threshold_stagnation
        self.best_genome = None
        self.best_genome_id = 0
        self.stagnation = 0

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        current_best_fitness = self.current_best_fitness
        best_genome_id = self.best_genome_id

        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (self.targets, genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
            if genome.fitness > self.current_best_fitness:
                current_best_fitness = genome.fitness
                self.best_genome = genome
                best_genome_id = ignored_genome_id

        if self.best_genome_id == best_genome_id and round(current_best_fitness, 2) <= round(self.current_best_fitness, 2):
            self.stagnation = self.stagnation + 1
        else:
            self.stagnation = 0

        self.current_best_fitness = current_best_fitness
        self.best_genome_id = best_genome_id

        if self.current_best_fitness > self.threshold_fitness and self.stagnation >= self.threshold_stagnation:
            print("CAMBIO TARGETS")
            self.targets = []
            for i in range(10):
                self.targets.append(tuple(np.random.uniform(0.1, 0.8, (1, 2))[0]))
            self.stagnation = 0


