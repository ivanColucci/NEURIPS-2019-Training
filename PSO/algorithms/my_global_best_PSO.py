import logging
import pickle
import time
# Import modules
import numpy as np
import multiprocessing as mp
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.reporter import Reporter

class MyGlobalBestPSO(GlobalBestPSO):
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
        # pool = None if n_processes is None else mp.Pool(n_processes)
        pool = mp.Pool(n_processes)
        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        for i in self.rep.pbar(iters, self.name):
            last_time = time.time()
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            # fmt: on
            self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            if (
                    np.abs(self.swarm.best_cost - best_cost_yet_found)
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
                "Generation {}: Mean-fitness {}, std_dev_fitness {}, best {}".format(i, np.mean(self.swarm.current_cost), np.std(self.swarm.current_cost), self.swarm.best_cost),
                lvl=logging.INFO,
            )
            self.rep.log("Time elapsed {}".format(time.time()-last_time), lvl=logging.INFO,)
            with open(filename, "wb") as fout:
                pickle.dump(self.swarm, fout)
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=logging.INFO,
        )
        return (final_best_cost, final_best_pos)