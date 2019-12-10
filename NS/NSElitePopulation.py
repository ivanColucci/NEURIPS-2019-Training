from neat.six_util import iteritems, itervalues
from neat.population import Population, CompleteExtinctionException
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NSElitePopulation(Population):

    def __init__(self, config, initial_state=None, overwrite=True, winner="", n_neighbors=5, novelty_threshold=0.5):
        super().__init__(config, initial_state)
        self.overwrite = overwrite
        self.winner_name = winner
        self.novelty_archive = []
        self.novelty_threshold = novelty_threshold
        self.n_neighbors = n_neighbors
        self.last_genome_added = None
        self.last_archive_modified = 0
        self.n_add_archive = 0

    def run(self, fitness_function, n=None):
        # Variables needed to save archive on each update
        winner_name_prefix = "archive_checkpoint_" + self.winner_name

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            not_evaluated = {}
            evaluated = []
            # len(population) = 2*pop_size
            for gid, g in self.population.items():
                if g.fitness is None:
                    not_evaluated[gid] = g
                else:
                    evaluated.append((gid, g))

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)
            self.KNNdistances(self.population, self.novelty_archive, self.n_neighbors)

            self.population = []
            self.population += evaluated
            for key, v in not_evaluated.items():
                self.population.append((key, v))
            self.population.sort(reverse=True, key=lambda x: x[1].dist)
            self.population = from_list_to_dict(self.population[:self.config.pop_size])

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
                if best is None or g.dist > best.dist:
                    best = g
                if g.dist > self.novelty_threshold:
                    if g not in self.novelty_archive:
                        self.novelty_archive.append(g)
                        print("Distanza di aggiunta: ", g.dist)
                        self.n_add_archive += 1
                        self.last_archive_modified = self.generation
                        with open(winner_name_prefix, "wb") as f:
                            pickle.dump(self.novelty_archive, f)

            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.dist for g in itervalues(self.population))
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

            time_diff = self.generation - self.last_archive_modified

            if time_diff > 60:
                self.novelty_threshold -= self.novelty_threshold * 0.05

            if self.n_add_archive > 4 and time_diff <= 30:
                self.novelty_threshold += self.novelty_threshold * 0.05
                self.n_add_archive = 0

            if time_diff > 30:
                self.n_add_archive = 0

            self.reporters.info("Novelty's archive size: {}\n".format(len(self.novelty_archive)))
            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.novelty_archive

    def KNNdistances(self, population, archive, n_neighbors):
        X = list()
        pop = [elem.fitness for elem in list(itervalues(population))]
        X.extend(pop)
        X.extend(elem.fitness for elem in archive)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, label='')
        # for elem in itervalues(population):
        #     ax.scatter(elem.fitness[0], elem.fitness[1], color='blue')
        # for elem in archive:
        #     ax.scatter(elem.fitness[0], elem.fitness[1], color='orange')
        # ax.set_xlabel('Distance')
        # # ax.set_ylabel('Mean Height Pelvis')
        # ax.set_ylabel('RMSE')
        # plt.title('K: ' + str(self.n_neighbors) + ', Th: ' + str(self.novelty_threshold))
        # plt.savefig('NS/Figures/Figure'+str(self.generation)+'.png')
        # plt.close(fig)

        X = np.array(X)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X)
        distances, _ = nbrs.kneighbors(np.array(pop))

        i = 0
        for key, elem in list(iteritems(population)):
            elem.dist = np.mean(distances[i])
            i += 1


def from_list_to_dict(l):
    d = {}
    for gid, g in l:
        d[gid] = g
    return d
