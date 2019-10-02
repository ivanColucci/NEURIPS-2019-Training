from PSO.PSO_problem import ball_catching
from mpi4py import MPI
import pickle
import pyswarms.single as algo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
prob = ball_catching()
bounds = prob.get_bounds()
dimension = prob.num_of_weights
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
n_islands = comm.Get_size()
n_iterations = 1
n_gen_island = 1
n_gen_master = 1
pop_size_island = 5
pop_size_master = n_islands

if rank == 0:
    import random
    import numpy as np
    random.seed(1234+rank)
    np.random.seed(1234+rank)
    # Generate initial population
    optimizer = algo.GlobalBestPSO(n_particles=pop_size_master, dimensions=dimension, options=options, bounds=bounds)
    temp = [0 for i in range(pop_size_master)]
    optimizer.swarm.current_cost = temp
    for i in range(n_iterations):
        print("*************** Generazione "+str(i+1)+": ***************")
        # when a slave ends, receive the best and put it in the population
        for j in range(1, n_islands):
            data = comm.recv(source=MPI.ANY_SOURCE)
            optimizer.swarm.current_cost[j] = data[0]
            optimizer.swarm.position[j] = data[1]
            print("from island", j, " : ", data[0])
        # when all the slaves are done, evolve the population
        best_f, best_x = optimizer.optimize(prob.fitness, iters=n_gen_master, n_processes=pop_size_master)
        if i == n_iterations - 1:
            # save the best individual
            with open("LAST", "wb") as fout:
                pickle.dump(best_x, fout)
            # stop slaves
            for j in range(1, n_islands):
                comm.send(0, dest=j, tag=200)
            break
        else:
            # Send the best individual to the islands
            print("Best fitness on master: " + str(best_f))
            data = (best_f, best_x)
            for j in range(1, n_islands):
                comm.send(data, dest=j)
        with open("champion_Parallel_gen"+str(i+1), "wb") as fout:
            pickle.dump(best_x, fout)
else:
    import random
    import numpy as np
    random.seed(1234 + rank)
    np.random.seed(1234 + rank)
    # initialize island
    optimizer = algo.GlobalBestPSO(n_particles=pop_size_island, dimensions=dimension, options=options, bounds=bounds)
    # evolve population
    best_f, best_x = optimizer.optimize(prob.fitness, iters=n_gen_island, n_processes=pop_size_island)
    # send best to master
    data = (best_f, best_x)
    comm.send(data, dest=0)
    while True:
        # wait for the best individual from master
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 200:
            break
        else:
            # replace a genome in the population
            pos = random.randint(0, len(optimizer.swarm_size)-1)
            optimizer.swarm.current_cost[pos] = data[0]
            optimizer.swarm.position[pos] = data[1]
        # evolve population
        best_f, best_x = optimizer.optimize(prob.fitness, iters=n_gen_island, n_processes=pop_size_island)
        # send best to master
        data = (best_f, best_x)
        comm.send(data, dest=0)
