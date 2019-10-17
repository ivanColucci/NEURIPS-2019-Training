import logging
import pickle
import time
# Import modules
import numpy as np
import multiprocessing as mp
# from pyswarms.backend.operators import compute_pbest, compute_objective_function
from PSO.algorithms.my_operators import compute_objective_function, compute_pbest
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.reporter import Reporter
from PSO.fitness_obj import FitnessObj

def to_arrays(vector):
    distance_array = []
    energy_array = []
    for el in vector:
        distance_array.append(el.distance)
        energy_array.append(el.energy)
    return distance_array, energy_array

def mean_vector(vector):
    d, e = to_arrays(vector)
    return (np.mean(d), np.mean(e))

def std_vector(vector):
    d, e = to_arrays(vector)
    return (np.std(d), np.std(e))

class MOPSO(GlobalBestPSO):
    def set_reporter_name(self, name):
        self.rep = Reporter(logger=logging.getLogger(name))

    def load_swarm(self, filename):
        with open(filename, "rb") as fin:
            self.swarm = pickle.load(fin)

    def optimize(self, objective_func, iters, n_processes=None, **kwargs):
        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=logging.INFO,
        )
        self.rep.log(
            "Population size: {}".format(self.swarm_size),
            lvl=logging.INFO,
        )
        filename = "CHECKPOINT_PYSWARMS_" + self.rep.logger.name
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)
        if type(self.swarm.best_cost) is float:
            self.swarm.best_cost = FitnessObj(0.0, np.inf)
            self.swarm.pbest_cost = np.array([FitnessObj(0.0, np.inf) for i in range(self.swarm_size[0])])
        for i in self.rep.pbar(iters, self.name):
            last_time = time.time()
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = np.array(compute_objective_function(self.swarm, objective_func, pool=pool))
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            # fmt: on
            self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=mean_vector(self.swarm.pbest_cost.tolist()),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + best_cost_yet_found.distance)
            if (
                    np.abs(self.swarm.best_cost.distance - best_cost_yet_found.distance)
                    < relative_measure
            ):
                break
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
            self.rep.log(
                "Generation {}: Mean-fitness {}, std_dev_fitness {}, best {}".format(i, mean_vector(self.swarm.current_cost.tolist()), std_vector(self.swarm.current_cost.tolist()), self.swarm.best_cost),
                lvl=logging.INFO,
            )
            self.rep.log("Time elapsed {}".format(time.time()-last_time), lvl=logging.INFO,)
            with open(filename, "wb") as fout:
                pickle.dump(self.swarm, fout)
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.__copy__()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=logging.INFO,
        )
        return (final_best_cost, final_best_pos)