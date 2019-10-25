from PSO.Problems.PSO_abstract_problem import WalkingProblem
import numpy as np
import math

class SOWalkingProblem(WalkingProblem):

    def fitness_hook(self, model, env):
        final_rew = 0
        observation = env.get_observation()
        #ivan
        pelvis_heights = []
        last_pelvis = env.get_state_desc()['body_pos']["pelvis"][0]
        pelvis_x = []
        for i in range(1, self.steps + 1):
            obs = np.reshape(observation, (1, 339, 1))
            action = model.predict_on_batch(obs)
            observation, reward, done, info = env.step(action[0], obs_as_dict=False)
            #ivan
            pelvis = env.get_state_desc()['body_pos']["pelvis"]
            pelvis_heights.append(pelvis[1])
            pelvis_x.append(pelvis[0] - last_pelvis)
            last_pelvis = pelvis[0]

            for elem in observation:
                if math.isnan(elem):
                    return [0.]
            if done:
                break
            final_rew += reward
        #ivan
        area = 0
        for i in range(len(pelvis_x)):
            area += pelvis_x[i] * pelvis_heights[i]
        return -area
        #return final_rew

    def reward_hook(self, env):
        env.set_reward_function(env.distance_reward)
        return env