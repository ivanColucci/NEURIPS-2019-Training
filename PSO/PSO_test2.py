import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from PSO.PSO_problem import ball_catching
import numpy as np

if __name__ == "__main__":
    with open("champion_pyswarms_conv_1", "rb") as fin:
        champion_x = pickle.load(fin)
        problem = ball_catching()
        x = []
        x.append(champion_x)
        result = problem.fitness(x)
        print(result)