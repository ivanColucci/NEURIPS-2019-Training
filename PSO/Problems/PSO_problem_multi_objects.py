from PSO.Problems.PSO_abstract_problem import WalkingProblem
import numpy as np
import math
from PSO.fitness_obj import FitnessObj

class MOWalkingProblem(WalkingProblem):

    def fitness_hook(self, model, env):
        total_distance = 0
        total_energy = 0
        observation = env.get_observation()
        for i in range(1, self.steps + 1):
            obs = np.reshape(observation, (1, 339, 1))
            action = model.predict_on_batch(obs)
            observation, reward, done, info = env.step(action[0], obs_as_dict=False)
            for elem in observation:
                if math.isnan(elem):
                    return [0.]
            if done:
                break
            total_energy += self.get_energy(env)
            total_distance += reward
        resultObj = FitnessObj(total_distance, total_energy)
        return resultObj


    def reward_hook(self, env):
        env.set_reward_function(env.distance_reward)
        return env