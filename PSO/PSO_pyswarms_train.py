# Import modules
import sys
import numpy as np
import pickle
from PSO.PSO_problem import WalkingProblem
import random
from PSO.my_global_best_PSO import MyGlobalBestPSO
from PSO.my_local_best_PSO import MyLocalBestPSO

random.seed(1234)
np.random.seed(1234)

def set_value(argv):
    n_gen = 200
    pop_size = 20
    g_best = False
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

    load_elem = True

    prob = WalkingProblem()
    bounds = prob.get_bounds()
    dimension = prob.num_of_weights
    # Set-up hyper-parameters & Call instance of PSO
    if g_best:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = MyGlobalBestPSO(n_particles=pop_size, dimensions=dimension, options=options, bounds=bounds)
    else:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        optimizer = MyLocalBestPSO(n_particles=pop_size, dimensions=dimension, options=options, bounds=bounds)
    if load_elem:
        with open("champion_pyswarms_conv_1","rb") as fin:
            best = pickle.load(fin)
            optimizer.swarm.position[0] = best
    # Perform optimization
    cost, pos = optimizer.optimize(prob.fitness_manager, iters=n_gen, n_processes=8)
    optimizer.set_reporter_name("output_PSO_conv1D_11/10.txt")
    print(cost)
    with open("champion_pyswarms_11_10", "wb") as fout:
        pickle.dump(pos, fout)
