from osim.env import L2M2019Env

class RewardShapingEnv(L2M2019Env):
    def __init__(self, visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None, reward_function=None):
        if reward_function is None:
            self.reward_function = self.get_reward_1()
        else:
            self.reward_function = reward_function
        super(RewardShapingEnv, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy, difficulty=difficulty, seed=seed, report=report)

    def get_reward(self):
        return self.reward_function()

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function