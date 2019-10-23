"""Implements the core evolution algorithm."""
from neat.six_util import iteritems, itervalues
from neat.population import Population, CompleteExtinctionException
import time
import pickle
import numpy as np


class TimePopulation(Population):

    def run(self, fitness_function, n=None):
        #Variables needed to save winner
        winner_interval = 1
        winnername_prefix = "winner_checkpoint_"
        last_winner_checkpoint = 0

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            start_time_gen = time.time()
            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            with open("output.txt", "a") as f:
                f.write("Gen: " + str(k) + " tempo: " + str(round(time.time() - start_time_gen,3)) + " sec\n")

            start_time_gen = time.time()
            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1
            #Code to save the best after winner_interval generations
            if winner_interval is not None:
                dg = self.generation - last_winner_checkpoint
                if dg >= winner_interval:
                    filename = '{0}{1}'.format(winnername_prefix, self.generation)
                    last_winner_checkpoint = self.generation
                    with open(filename, 'wb') as f:
                        pickle.dump(self.best_genome, f)

            with open("output.txt", "a") as f:
                f.write("Gen: " + str(k) + " tempo: " + str(round(time.time() - start_time_gen,3)) + " sec\n")

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
