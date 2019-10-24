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

    def distance_and_energy(self):
        return 100*self.distance_reward() - self.energy_consumption_reward()/22

    def get_observation(self):
        obs_dict = self.get_observation_dict()
        res = []

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res