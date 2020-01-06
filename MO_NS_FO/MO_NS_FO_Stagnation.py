import sys

from neat.six_util import iteritems
from neat.stagnation import DefaultStagnation


class Stagnation(DefaultStagnation):

    def __init__(self, config, reporters):
        super().__init__(config, reporters)

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
        species_data = []
        for sid, s in iteritems(species_set.species):
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            if s.dist_history:
                prev_dist = max(s.dist_history)
            else:
                prev_dist = -sys.float_info.max

            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            s.dist = self.species_fitness_func(s.get_distances())
            s.dist_history.append(s.fitness)
            s.adjusted_dist = None
            if prev_dist is None or s.dist > prev_dist:
                s.last_improved = generation

            s.rank = max(s.get_rank())
            species_data.append((sid, s))

        # Sort in ascending rank order.
        species_data.sort(reverse=True, key=lambda x: x[1].rank)

        result = []
        # species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            # species_fitnesses.append(s.dist)

        return result
