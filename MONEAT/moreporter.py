import time
from neat.reporting import BaseReporter
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
from MONEAT.fitness_obj import mean_vector, std_vector, FitnessObj

class MOReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail, filename):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.filename = filename

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        with open(self.filename, "a+") as f:
            f.write('\n ****** Running generation {0} ****** \n\n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        with open(self.filename, "a+") as f:
            if self.show_species_detail:
                print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
                f.write('Population of {0:d} members in {1:d} species:\n'.format(ng, ns))
                sids = list(iterkeys(species_set.species))
                sids.sort()
                print("   ID   age  size       fitness           adj fit      stag")
                print("  ====  ===  ====  ================  ================  ====")
                f.write("   ID   age  size       fitness           adj fit      stag\n")
                f.write("  ====  ===  ====  ================  ================  ====\n")
                for sid in sids:
                    s = species_set.species[sid]
                    a = self.generation - s.created
                    n = len(s.members)
                    fi = "--" if s.fitness is None else "{}".format(round(s.fitness, 1))
                    af = "--" if s.adjusted_fitness is None else "{}".format(round(s.adjusted_fitness, 3))
                    st = self.generation - s.last_improved
                    print(
                        "  {: >4}  {: >3}  {: >4}  {: >16}  {: >16}  {: >4}".format(sid, a, n, fi, af, st))
                    f.write(
                        "  {: >4}  {: >3}  {: >4}  {: >16}  {: >16}  {: >4}\n".format(sid, a, n, fi, af, st))
            else:
                print('Population of {0:d} members in {1:d} species'.format(ng, ns))
                f.write('Population of {0:d} members in {1:d} species\n'.format(ng, ns))

            elapsed = time.time() - self.generation_start_time
            self.generation_times.append(elapsed)
            self.generation_times = self.generation_times[-10:]
            average = sum(self.generation_times) / len(self.generation_times)
            f.write('Total extinctions: {0:d}\n'.format(self.num_extinctions))
            if len(self.generation_times) > 1:
                f.write("Generation time: {0:.3f} sec ({1:.3f} average)\n".format(elapsed, average))
            else:
                f.write("Generation time: {0:.3f} sec\n".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        if type(fitnesses[0]) is float:
            fit_mean = mean(fitnesses)
            fit_std = stdev(fitnesses)
        elif type(fitnesses[0]) is FitnessObj:
            fit_mean = mean_vector(fitnesses)
            fit_std = std_vector(fitnesses)
        else:
            fit_mean = 0
            fit_std = 0
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        with open(self.filename, "a+") as f:
            f.write('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}\n'.format(fit_mean, fit_std))
            f.write(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}\n'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')
        with open(self.filename, "a+") as f:
            f.write('All species extinct.\n')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))
        with open(self.filename, "a+") as f:
            f.write('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}\n'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))
            with open(self.filename, "a+") as f:
                f.write("\nSpecies {0} with {1} members is stagnated: removing it\n".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)
        with open(self.filename, "a+") as f:
            f.write(msg)