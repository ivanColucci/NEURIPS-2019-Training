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
        
    def standard_reward(self):
        state_desc = self.get_state_desc()
        if not self.get_prev_state_desc():
            return 0

        reward = 0
        dt = self.osim_model.stepsize

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        reward += self.d_reward['alive']

        # effort ~ muscle fatigue ~ (muscle activation)^2
        ACT2 = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
        self.d_reward['effort'] += ACT2*dt
        self.d_reward['footstep']['effort'] += ACT2*dt

        self.d_reward['footstep']['del_t'] += dt

        # reward from velocity (penalize from deviating from v_tgt)

        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = self.vtgt.get_vtgt(p_body).T

        self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        # footstep reward (when made a new step)
        if self.footstep['new']:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
            reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0

            # panalize effort
            reward_footstep_e = -self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

            self.d_reward['footstep']['del_t'] = 0
            self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0

            reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        # success bonus
        if not self.is_done() and (self.osim_model.istep >= self.spec.timestep_limit): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            #reward += reward_footstep_0 + 100
            reward += 10

        return reward

    def state_desc_to_list(self):
        state_desc = self.get_state_desc()
        res = []

        # Body Observations
        for info_type in ['body_pos', 'body_pos_rot',
                          'body_vel', 'body_vel_rot',
                          'body_acc', 'body_acc_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l', 'femur_l',
                              'calcn_r', 'talus_r', 'tibia_r', 'toes_r', 'femur_r',
                              'head', 'pelvis', 'torso']:
                res += state_desc[info_type][body_part]

        # Joint Observations
        # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
        for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += state_desc[info_type][joint]



        # Muscle Observations
        for muscle in ['abd_l', 'abd_r',
                       'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r',
                       'gastroc_l', 'gastroc_r',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r',
                       'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'soleus_l',
                       'tib_ant_l', 'tib_ant_l',
                       'vasti_l', 'vasti_r']:
            res.append(state_desc['muscles'][muscle]['activation'])
            res.append(state_desc['muscles'][muscle]['fiber_force'])
            res.append(state_desc['muscles'][muscle]['fiber_length'])
            res.append(state_desc['muscles'][muscle]['fiber_velocity'])

        # Force Observations
        # Neglecting forces corresponding to muscles as they are redundant with
        # `fiber_forces` in muscles dictionaries
        for force in ['AnkleLimit_l', 'AnkleLimit_r',
                      'HipAddLimit_l', 'HipAddLimit_r',
                      'HipLimit_l', 'HipLimit_r',
                      'KneeLimit_l', 'KneeLimit_r',
                      'foot_l', 'foot_r']:
            res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

        return res
