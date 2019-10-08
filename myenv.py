from osim.env import L2M2019Env
import numpy as np

class RewardShapingEnv(L2M2019Env):
    def __init__(self, visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None, reward_function=None):
        self.prev_distance = 0
        if reward_function is None:
            self.reward_function = self.get_reward_1
        else:
            self.reward_function = reward_function
        super(RewardShapingEnv, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy, difficulty=difficulty, seed=seed, report=report)

    def get_reward(self):
        return self.reward_function()

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def keep_alive_reward(self):
        return 0.1

    def distance_reward(self):
        state = self.get_state_desc()
        current_distance = state['body_pos']['pelvis'][0]
        diff = current_distance - self.prev_distance
        self.prev_distance = current_distance
        return diff

    def energy_consumption_reward(self):
        state_desc = self.get_state_desc()
        ACT2 = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
        return ACT2

    def step_accuracy_reward(self):
        pass