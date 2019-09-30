# Import modules
import sys
import numpy as np
import pickle
from PSO_problem import ball_catching
import random
import pyswarms.single as algo

random.seed(1234)
np.random.seed(1234)

def set_value(argv):
    n_gen = 100
    pop_size = 10
    g_best = True
    if len(argv) > 1:
        n_gen = int(sys.argv[1])
        pop_size = int(sys.argv[2])
        g_best = int(sys.argv[3]) == 1
    return n_gen, pop_size, g_best


if __name__ == "__main__":
    # argv: 1) gen 2) pop_size 3) global==1 local==2
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    n_gen, pop_size, g_best = set_value(sys.argv)

    prob = ball_catching()
    bounds = prob.get_bounds()
    dimension = prob.num_of_weights
    # Set-up hyper-parameters & Call instance of PSO
    if g_best:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = algo.GlobalBestPSO(n_particles=pop_size, dimensions=dimension, options=options, bounds=bounds)
    else:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        optimizer = algo.LocalBestPSO(n_particles=pop_size, dimensions=dimension, options=options, bounds=bounds)
    # Perform optimization
    cost, pos = optimizer.optimize(prob.fitness, iters=n_gen, n_processes=pop_size)
    print(cost)
    with open("champion_pyswarms", "wb") as fout:
        pickle.dump(pos, fout)
