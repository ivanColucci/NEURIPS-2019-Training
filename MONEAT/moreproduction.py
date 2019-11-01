from __future__ import division

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

from NEAT.my_reproduction import TournamentReproduction
from MONEAT.fitness_obj import mean_vector, sum_vector, FitnessObj, split_genomes


class MOReproduction(TournamentReproduction):

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    @staticmethod
    def compute_adjusted_fitness(all_fitnesses, remaining_species):
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(FitnessObj(0.1, 1), max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            array = [m.fitness for m in itervalues(afs.members)]
            if type(array[0]) is FitnessObj:
                msf = mean_vector(array)
            else:
                msf = mean(array)
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        if type(adjusted_fitnesses[0]) is FitnessObj:
            avg_adjusted_fitness = mean_vector(adjusted_fitnesses)
        else:
            avg_adjusted_fitness = mean(adjusted_fitnesses)
        return adjusted_fitnesses, avg_adjusted_fitness

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        if type(adjusted_fitness[0]) is FitnessObj:
            af_sum = sum_vector(adjusted_fitness)
        else:
            af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            if type(d) is FitnessObj:
                c = int(round(d.distance))
            else:
                c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def elitism_politic(self, old_members, new_population, spawn):
        pareto, dominated = split_genomes(old_members)
        for i, m in pareto:
            new_population[i] = m
            spawn -= 1
