from __future__ import division
import math
import random
from neat.math_util import mean
from neat.reproduction import DefaultReproduction
from neat.six_util import iteritems, itervalues
from NS.NSElitePopulation import from_list_to_dict
random.seed(1234)


class NSEliteReproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.tournament_threshold = 0.1

    @staticmethod
    def tournament(members, tournament_threshold):
        spec_size = len(members)
        num_of_element = max(2, round(tournament_threshold * spec_size))
        selected = random.sample(members, k=num_of_element)
        selected.sort(reverse=True, key=lambda x: x[1].dist)
        parent1_id, parent1 = selected[:1][0]
        parent2_id = parent1_id
        parent2 = parent1
        while parent2_id == parent1_id:
            selected = random.sample(members, k=num_of_element)
            selected.sort(reverse=True, key=lambda x: x[1].dist)
            parent2_id, parent2 = selected[:1][0]

        return parent1_id, parent1, parent2_id, parent2

    def reproduce(self, config, species, pop_size, generation):
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.dist for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)

        if not remaining_species:
            species.species = {}
            return {}

        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)

        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.dist for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size

        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        all_old_members = []
        species.species = {}

        population = []
        for s in remaining_species:
            population += list(iteritems(s.members))

        for spawn, s in zip(spawn_amounts, remaining_species):

            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            old_members = list(iteritems(s.members))
            all_old_members += old_members
            s.members = {}
            species.species[s.key] = s

            old_members.sort(reverse=True, key=lambda x: x[1].dist)

            if spawn <= 0:
                continue

            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            while spawn > 0:
                spawn -= 1

                if len(old_members) > 2:
                    parent1_id, parent1, parent2_id, parent2 = self.tournament(old_members, self.tournament_threshold)
                else:
                    parent1_id, parent1 = random.choice(old_members)
                    parent2_id, parent2 = random.choice(old_members)

                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        all_old_members.sort(key=lambda x: x[0])
        all_old_members = from_list_to_dict(all_old_members)
        new_population.update(all_old_members)

        return new_population

