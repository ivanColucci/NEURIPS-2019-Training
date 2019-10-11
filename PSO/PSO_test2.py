import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from PSO.PSO_problem import WalkingProblem
import numpy as np

if __name__ == "__main__":
    with open("champion_pyswarms", "rb") as fin:
        champion_x = pickle.load(fin)
        problem = WalkingProblem()
        x = []
        x.append(champion_x)
        result = problem.fitness(x)
        print(result)