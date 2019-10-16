import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from PSO.Problems.PSO_abstract_problem import WalkingProblem

if __name__ == "__main__":
    with open("champion_pyswarms_conv_1", "rb") as fin:
        champion_x = pickle.load(fin)
        problem = WalkingProblem()
        x = []
        x.append(champion_x)
        result = problem.fitness(x)
        print(result)