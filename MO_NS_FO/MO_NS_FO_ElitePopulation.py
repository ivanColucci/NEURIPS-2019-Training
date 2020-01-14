import pickle
import math
import numpy as np
from neat.population import Population, CompleteExtinctionException
from neat.six_util import iteritems, itervalues
from sklearn.neighbors import NearestNeighbors

from NEAT.utils.utilities import from_list_to_dict
import matplotlib.pyplot as plt

class ElitePopulation(Population):

    def __init__(self, config, initial_state=None, overwrite=True, winner="", n_neighbors=5, novelty_threshold=0.5, use_archive=False):
        super().__init__(config, initial_state)
        self.overwrite = overwrite
        self.winner_name = winner
        self.n_neighbors = n_neighbors
        self.last_genome_added = None
        self.last_archive_modified = 0
        self.n_add_archive = 0
        self.novelty_archive = []
        self.use_archive = use_archive
        self.novelty_threshold = novelty_threshold

    def run(self, fitness_function, n=None):
        # Variables needed to save archive on each update
        winner_name_prefix = "front_" + self.winner_name

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            front = []
            self.reporters.start_generation(self.generation)

            not_evaluated = {}
            evaluated = []
            # len(population) = 2*pop_size
            for gid, g in self.population.items():
                g.rank = None
                if g.fitness is None:
                    not_evaluated[gid] = g
                else:
                    evaluated.append((gid, g))
                if self.use_archive and g not in self.novelty_archive and g.dist is not None and g.dist > self.novelty_threshold:
                    self.novelty_archive.append(g)
                    self.n_add_archive += 1
                    self.last_archive_modified = self.generation
                    with open("archive_"+self.winner_name, "wb") as f:
                        pickle.dump(self.novelty_archive, f)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(not_evaluated)), self.config)
            # calculate distance on 2*pop_size
            if self.use_archive:
                self.KNNdistances(self.population, self.n_neighbors, archive=self.novelty_archive)
            else:
                self.KNNdistances(self.population, self.n_neighbors)
            self.calculateDifferentRanks()

            # fig = plt.figure()
            # ax = fig.add_subplot(111, label='')
            # for elem in self.population.items():
            #     if elem[1].rank == 0:
            #         front.append(elem)
            #         ax.scatter(elem[1].fitness, elem[1].dist, color='orange')
            #         ax.annotate(elem[1].rank, (elem[1].fitness, elem[1].dist))
            #     else:
            #         ax.scatter(elem[1].fitness, elem[1].dist, color='blue')
            #         ax.annotate(elem[1].rank, (elem[1].fitness, elem[1].dist))
            #
            # ax.set_xlabel('Fitness')
            # # ax.set_ylabel('Mean Height Pelvis')
            # ax.set_ylabel('Mean Diversity')
            # plt.axis((0, plt.axis()[1], 0, plt.axis()[3]))
            # plt.title('MO_NS_FO: POP=' + str(len(self.population)))
            # plt.savefig('MO_NS_FO/Figures/Figure'+str(self.generation)+'.png')
            # plt.close(fig)

            population = []
            for gid, g in self.population.items():
                population.append((gid, g))
            population.sort(reverse=False, key=lambda x: x[1].rank)
            self.population = population[:self.config.pop_size]
            self.population = from_list_to_dict(self.population)
            self.species.speciate(self.config, self.population, self.generation)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g

            with open(winner_name_prefix, "wb") as f:
                pickle.dump(front, f)

            self.reporters.post_evaluate(self.config, self.population, self.species, best)

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

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.dist for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            self.reporters.end_generation(self.config, self.population, self.species)

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

            if self.use_archive:
                time_diff = self.generation - self.last_archive_modified

                if time_diff > 60:
                    self.novelty_threshold -= self.novelty_threshold * 0.05

                if self.n_add_archive > 4 and time_diff <= 30:
                    self.novelty_threshold += self.novelty_threshold * 0.05
                    self.n_add_archive = 0

                if time_diff > 30:
                    self.n_add_archive = 0

                self.reporters.info("Novelty's archive size: {}\n".format(len(self.novelty_archive)))
            self.reporters.info("Front size: {}\n".format(len(front)))
            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return front

    def KNNdistances(self, population, n_neighbors, archive=None):
        pop = []
        for elem in list(itervalues(population)):
            if elem.phenotype is not None:
                pop.append(elem.phenotype)
            else:
                pop.append([0, 0])

        if archive is not None:
            for elem in archive:
                if elem.phenotype is not None:
                    pop.append(elem.phenotype)
                else:
                    pop.append([0, 0])

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(np.array(pop))

        distances, _ = nbrs.kneighbors(np.array(pop))

        i = 0
        for key, elem in list(iteritems(population)):
            elem.dist = np.mean(distances[i])
            i += 1

    def computeRank(self, population):
        for g1 in itervalues(population):
            if g1.rank is None:
                for g2 in itervalues(population):
                    if g1.is_dominant(g2):
                        g2.rank = 1
                    if g2.is_dominant(g1):
                        g1.rank = 1
                        break
            if g1.rank is None:
                g1.rank = 2

    def calculateDifferentRanks(self):
        dominated_dict = self.population
        flag = True
        current_rank = 0
        dominated = []

        while len(dominated_dict) != 0 and flag:
            flag = False
            self.computeRank(dominated_dict)
            for gid, g in dominated_dict.items():
                if g.rank == 2:
                    flag = True
                    g.rank = current_rank
                else:
                    g.rank = None
                    dominated.append((gid, g))
            dominated_dict = from_list_to_dict(dominated)
            dominated = []
            current_rank += 1
