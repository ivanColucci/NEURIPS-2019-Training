from __future__ import division

import math
import random
from itertools import count
import numpy as np
from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from NEAT.utils.parallel_creation import ParallelCreator
from NEAT.utils.utilities import print_file

class TournamentReproduction(DefaultClassConfig):

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    def __init__(self, config, reporters, stagnation, rigeneration=False):
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        self.rigeneration = rigeneration
        self.last_rigeneration = 0

    def allow_rigeneration(self, value):
        self.rigeneration = value

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = self.create_new_parallel(genome_type, genome_config, num_genomes)
        return new_genomes

    def create_new_parallel(self, genome_type, genome_config, num_genomes, random_hidden=False):
        pc = ParallelCreator(num_genomes, random_hidden=random_hidden)
        new_genomes = pc.create_new(self, genome_type, genome_config)
        del pc
        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
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

    def reproduce(self, config, species, pop_size, generation):
        all_fitnesses = []
        remaining_species = []
        num_stagnant_genomes = 0
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
                num_stagnant_genomes += len(stag_s.members)
            else:
                stagnant_time = generation - stag_s.last_improved
                if stagnant_time > self.stagnation.stagnation_config.max_stagnation:
                    num_stagnant_genomes += len(stag_s.members)
                all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)

        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        min_species_size = max(min_species_size,self.reproduction_config.elitism)

        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))

            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # spawn = Pop_size - elit.
            # Pop_size - num_stagnant_genomes == evoluzione
            # num_stagnant_genomes == da rimpiazzare con genomi freschi
            if self.rigeneration and num_stagnant_genomes > 0 \
                    and ((generation - self.last_rigeneration) >= self.stagnation.stagnation_config.max_stagnation):

                self.last_rigeneration = generation
                prev_spawn = spawn
                num_stagnant_genomes = np.min([num_stagnant_genomes, spawn])
                new_genomes = self.get_new_genomes(num_stagnant_genomes, config, random_for_all=True)
                for gid, genome in new_genomes.items():
                    new_population[gid] = genome
                spawn -= len(new_genomes)
                print_file("\nprev spawn: " + str(prev_spawn) + "\n")
                print_file("adding: " + str(len(new_genomes)) + " genomes" + "\n")
                print_file("new spawn: " + str(spawn) + "\n")

            tournament_threshold = 0.2
            while spawn > 0:
                spawn -= 1

                if len(old_members) >= 2:
                    parent1_id, parent1, parent2_id, parent2 = self.tournament(old_members, tournament_threshold)
                else:
                    parent1_id, parent1 = random.choice(old_members)
                    parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                # print_file("p1 nodes: " + str(len(parent1.nodes)) + " p2 nodes: " + str(len(parent2.nodes)) + " ", file="prova.txt")
                child.configure_crossover(parent1, parent2, config.genome_config)
                # print_file("child nodes: " + str(len(child.nodes)) + "\n", file="prova.txt")
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population

    def get_new_genomes(self, num_stagnant_genomes, config, random_for_all=True):
        threshold = 0.3
        num_of_new_genomes = int(math.ceil(threshold * num_stagnant_genomes))

        if not random_for_all:
            prev_hidden = config.genome_config.num_hidden
            n_hidden = np.random.randint(prev_hidden + 1, 2 * prev_hidden)
            config.genome_config.num_hidden = n_hidden

        new_genomes = self.create_new_parallel(config.genome_type, config.genome_config, num_of_new_genomes, random_hidden=random_for_all)
        return new_genomes

    def tournament(self, members, tournament_threshold):
        "riceve il survival_threshold % della popolazione, restituisce 2 parent"
        spec_size = len(members)
        num_of_element = round(tournament_threshold*spec_size)
        selected = random.sample(members, k=num_of_element)
        selected.sort(reverse=True, key=lambda x: x[1].fitness)
        parent1_id, parent1 = selected[:1][0]
        parent2_id = parent1_id
        parent2 = parent1
        while parent2_id == parent1_id:
            selected = random.sample(members, k=num_of_element)
            selected.sort(reverse=True, key=lambda x: x[1].fitness)
            parent2_id, parent2 = selected[:1][0]

        return parent1_id, parent1, parent2_id, parent2