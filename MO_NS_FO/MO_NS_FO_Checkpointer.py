"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random

from MO_NS_FO.MO_NS_FO_ElitePopulation import ElitePopulation
from MO_NS_FO.MO_NS_FO_Population import Population

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

from neat.reporting import BaseReporter


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """
    def __init__(self, checkpoint_interval=100, filename_prefix='neat-checkpoint-', overwrite=False):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.checkpoint_interval = checkpoint_interval
        self.filename_prefix = filename_prefix
        self.overwrite = overwrite
        self.current_generation = None
        self.last_generation_checkpoint = -1

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        if self.checkpoint_interval is not None:
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.checkpoint_interval:
                self.save_checkpoint(config, population, species_set, self.current_generation)
                self.last_generation_checkpoint = self.current_generation

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        if self.overwrite:
            filename = self.filename_prefix
        else:
            filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename, elite=True, archive_name="archive_", n_neighbors=15, use_archive=True, winner=""):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)

            if elite:
                pop = ElitePopulation(config, (population, species_set, generation), n_neighbors=n_neighbors,
                                      use_archive=use_archive, winner=winner)
            else:
                pop = Population(config, (population, species_set, generation))
            with open(archive_name, "rb") as fp:
                pop.novelty_archive = pickle.load(fp)
            return pop
