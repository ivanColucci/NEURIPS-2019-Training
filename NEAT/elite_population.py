from neat.six_util import iteritems, itervalues
from neat.population import Population, CompleteExtinctionException
from NEAT.utils.utilities import print_file, from_list_to_dict
import time
import pickle


class ElitePopulation(Population):

    def __init__(self, config, initial_state=None, random_replace=False):
        super().__init__(config, initial_state)
        self.random_replace = random_replace

    def allow_regeneration(self, value):
        self.reproduction.allow_regeneration(value)

    def run(self, fitness_function, n=None):
        # Variables needed to save winner
        winner_interval = 10
        winner_name_prefix = "winner_checkpoint_"
        last_winner_checkpoint = 0

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            self.reporters.start_generation(self.generation)
            start_time_gen = time.time()

            # Evaluate all genomes using the user-provided function.
            not_evaluated = {}
            evaluated = []
            # len(population) = 2*pop_size
            for gid, g in self.population.items():
                if g.fitness is None:
                    not_evaluated[gid] = g
                else:
                    evaluated.append((gid, g))

            fitness_function(list(iteritems(not_evaluated)), self.config)

            if self.random_replace:
                i = 0
                self.population = {}
                for gid, g in not_evaluated.items():
                    if len(evaluated) <= i or g.fitness > evaluated[i][1].fitness:
                        self.population[gid] = g
                    else:
                        self.population[evaluated[i][0]] = evaluated[i][1]
                    i = i + 1
            else:
                self.population = []
                self.population += evaluated
                for k, v in not_evaluated.items():
                    self.population.append((k, v))
                self.population.sort(reverse=True, key=lambda x: x[1].fitness)
                self.population = from_list_to_dict(self.population[:self.config.pop_size])

            print_file("Gen: " + str(k) + " tempo: " + str(round(time.time() - start_time_gen,3)) + " sec\n")

            start_time_gen = time.time()

            self.species.speciate(self.config, self.population, self.generation)

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

            self.reporters.end_generation(self.config, self.population, self.species)

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            self.generation += 1
            # Code to save the best after winner_interval generations
            if winner_interval is not None:
                dg = self.generation - last_winner_checkpoint
                if dg >= winner_interval:
                    filename = '{0}{1}'.format(winner_name_prefix, self.generation)
                    last_winner_checkpoint = self.generation
                    with open(filename, 'wb') as f:
                        pickle.dump(self.best_genome, f)

            print_file("\nGen: " + str(k) + " tempo: " + str(round(time.time() - start_time_gen,3)) + " sec\n")

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
