from PSO_problem import ball_catching
from mpi4py import MPI
import pygmo as pg
import pickle
import random
random.seed(1234)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
prob = pg.problem(ball_catching())
n_iterations = 10
n_islands = comm.Get_size()
n_gen_island = 5
pop_size_island = 5

if rank == 0:
    print("Master")
    algo = pg.algorithm(pg.pso(1, seed=1234))
    algo.set_verbosity(1)
    pop = pg.population(prob=prob, size=n_islands, seed=1234)
    for i in range(n_iterations):
        print("Generazione "+str(i+1)+":")
        # when a slave ends, receive the best and put it in the population
        for j in range(1, n_islands):
            data = comm.recv(source=MPI.ANY_SOURCE)
            pop.set_xf(j, data[1], data[0])
        # when all the slaves are done, evolve the population
        pop = algo.evolve(pop)
        # send the best element of the current population to the slaves
        data = (pop.champion_f, pop.champion_x)
        print("Best fitness: "+str(pop.champion_f))
        for j in range(1, n_islands):
            comm.send(data, dest=j)
    #stop slaves
    data = (None, None)
    for j in range(1, n_islands):
        comm.send(data, dest=j)
    with open("championPSO_Parallel2", "wb") as fout:
        pickle.dump(pop.champion_x, fout)
else:
    print("Slave")
    # initialize island
    algo = pg.algorithm(pg.pso(n_gen_island, seed=1234+rank))
    algo.set_verbosity(1)
    pop = pg.population(prob, size=pop_size_island, seed=1234+rank)
    # evolve population
    pop = algo.evolve(pop)
    # send best to master
    data = (pop.champion_f, pop.champion_x)
    comm.send(data, dest=0)
    while(True):
        # wait for the best individual from master
        data = comm.recv(source=0)
        if data[0] is None:
            break
        #replace a genome in the population
        pop.set_xf(random.randint(0, len(pop)-1), data[1], data[0])
        #evolve population
        pop = algo.evolve(pop)
        # send best to master
        data = (pop.champion_f, pop.champion_x)
        comm.send(data, dest=0)





