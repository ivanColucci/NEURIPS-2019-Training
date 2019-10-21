import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from PSO.Problems.PSO_abstract_problem import WalkingProblem
from PSO.Problems.PSO_problem_single_object import SOWalkingProblem

if __name__ == "__main__":
    with open("../champion_pyswarms_16_10", "rb") as fin:
        champion_x = pickle.load(fin)
        problem = SOWalkingProblem()
        x = []
        x.append(champion_x)
        result = problem.fitness(x)
        print(result)