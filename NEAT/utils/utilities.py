import numpy as np
from myenv import RewardShapingEnv
import neat
from MONEAT.fitness_obj import FitnessObj
import pickle
import os
from NEAT.utils.rmse import calculate_sin, array_diff, rmse
MEAN_H = 0.9
MAX_H = 0.95
MIN_H = 0.85

INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01])

class Evaluator():
    #The reward_type parameter allows to choose the reward function to use.
    #Legend for reward type:
    #   1 - distance metric,
    #   2 - area metric,
    #   3 - step reward with a bonus for staying with the pelvis between 0.84 and 0.94,
    #   4 - step reward,
    #   any other value - use the reward_function passed by argument
    # old_inputs set to True to use 339 inputs (state + v_tgt), to False to use 97 inputs (only state)
    def __init__(self, reward_type=1, save_simulation=False, load_simulation=False, file_to_load="actions",
                 reward_function=None, old_input=True, visual=False, is_a_net=False, steps=500):
        self.reward_type = reward_type
        self.reward_function = reward_function
        self.old_input = old_input
        self.visual = visual
        self.is_a_net = is_a_net
        self.steps = steps
        self.load_simulation = load_simulation
        self.save_simulation = save_simulation
        self.file_name = file_to_load
        if self.load_simulation:
            local_dir = os.path.dirname(__file__)
            with open(os.path.join(local_dir, self.file_name), 'rb') as f:
                self.action_arr = pickle.load(f)

    @staticmethod
    def add_action_for_3d(action):
        Fmax_ABD = 4460.290481
        Fmax_ADD = 3931.8
        r_leg, l_leg = action[:9], action[9:]
        full_action = []
        full_action.append(0.1)
        full_action.append(0.1*Fmax_ADD/Fmax_ABD)
        for el in r_leg:
            full_action.append(el)
        full_action.append(0.1)
        full_action.append(0.1 * Fmax_ADD / Fmax_ABD)
        for el in l_leg:
            full_action.append(el)
        return full_action

    def load_next_action(self, i):
        return self.action_arr[i]

    def execute_trial(self, env, net, steps):
        final_rew = 0
        action_arr = []
        observation = env.get_observation()
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            final_rew += reward
            if done:
                break
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        return (10000 - final_rew)/1000 + 10*env.get_state_desc()['body_pos']["pelvis"][0]

    def execute_trial_with_distance(self, env, net, steps):
        observation = env.get_observation()
        action_arr = []
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            if done:
                break
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        return env.get_state_desc()['body_pos']["pelvis"][0]

    def execute_trial_with_body_in_range_distance(self, env, net, steps):
        observation = env.get_observation()
        action_arr = []
        total_reward = 0.0
        for i in range(steps):
            body = env.get_state_desc()['body_pos']
            # height
            h = body['pelvis'][1]
            # distance
            d = body['pelvis'][0]
            # reward
            total_reward += gaussian(h)*(1.0 * d)
            # save & load block
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            # submit action
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            if done:
                break
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)

        return total_reward

    def execute_trial_with_area(self, env, net, steps):
        final_rew = 0
        observation = env.get_observation()
        action_arr = []
        pelvis_heights = []
        last_pelvis = env.get_state_desc()['body_pos']["pelvis"][0]
        pelvis_x = []
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            pelvis = env.get_state_desc()['body_pos']["pelvis"]
            pelvis_heights.append(pelvis[1])
            pelvis_x.append(pelvis[0] - last_pelvis)
            last_pelvis = pelvis[0]
            final_rew += reward
            if done:
                break
        area = 0
        for i in range(len(pelvis_x)):
            h = pelvis_heights[i]
            if h < MIN_H:
                final_h = h
            elif h > MAX_H:
                final_h = MEAN_H - (h - MEAN_H)
            else:
                final_h = MEAN_H
            area += pelvis_x[i] * final_h
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        return area

    @staticmethod
    def get_reward(body_y, step_posx):
        dim = len(step_posx)
        total_rew = 0
        for h in body_y:
            if h < 0.8 or h > 0.95:
                total_rew += 0.1
        positions = []
        for i in range(dim):
            if i == 0:
                positions.append(0.4)
            else:
                positions.append(round(positions[-1] + 0.7, 1))
        diff = np.abs(np.subtract(step_posx, positions))
        clip_diff = np.clip(diff, a_min=0.0, a_max=0.7)
        rew = 0.7 - clip_diff
        total_rew += 100 * np.sum(rew)
        return total_rew

    @staticmethod
    def get_reward_h(body_y, step_posx):
        dim = len(step_posx)
        positions = []
        total_rew = 0

        for i in range(dim):
            if i == 0:
                positions.append(0.4)
            else:
                positions.append(round(positions[-1] + 0.7, 1))
        for i in range(dim):
            h = body_y[i]
            alfa = 1
            if h < 0.84:    # out of range
                alfa = h
            if h > 0.94:    # out of range
                alfa = 1 - h + 0.94
            total_rew += alfa * (0.7 - np.clip(np.abs(step_posx[i] - positions[i]), a_min=0.0, a_max=0.7))
        return total_rew

    def execute_trial_step_reward(self, env, net, steps):
        observation = env.get_observation()
        body_y = []
        step_posx = []
        action_arr = []
        doing_stepr = False
        doing_stepl = False
        step_threshold = 0.02
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            posy_r = env.get_state_desc()["body_pos"]["toes_r"][1]
            posy_l = env.get_state_desc()["body_pos"]["toes_l"][1]

            if self.reward_type == 3:
                body_y.append(env.get_state_desc()["body_pos"]["pelvis"][1])

            if posy_r < step_threshold and not doing_stepr:
                doing_stepr = True
                step_posx.append(env.get_state_desc()["body_pos"]["toes_r"][0])
                if self.reward_type == 4:
                    body_y.append(env.get_state_desc()["body_pos"]["pelvis"][1])
            if posy_l < step_threshold and not doing_stepl:
                doing_stepl = True
                step_posx.append(env.get_state_desc()["body_pos"]["toes_l"][0])
                if self.reward_type == 4:
                    body_y.append(env.get_state_desc()["body_pos"]["pelvis"][1])

            if doing_stepr and posy_r > step_threshold:
                doing_stepr = False
            if doing_stepl and posy_l > step_threshold:
                doing_stepl = False
            if done:
                break
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        if self.reward_type == 3:
            return self.get_reward(body_y, step_posx)
        elif self.reward_type == 4:
            return self.get_reward_h(body_y, step_posx)

    def execute_trial_progressive_reward(self, env, net, steps):
        observation = env.get_observation()
        left = []
        right = []
        action_arr = []
        posture_rew = 0
        objective_distance = 10.0
        objective_variance = 60
        objective_rmse = 0.1
        objective_posture = 7.0
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            rf = env.get_state_desc()["body_pos"]["toes_r"][0]
            lf = env.get_state_desc()["body_pos"]["toes_l"][0]
            curr_torso = env.get_body_com("torso")[0]
            l1_x = env.get_observation_dict()['l_leg']['knee'][0]
            l2_x = env.get_observation_dict()['r_leg']['knee'][0]
            legs = [l1_x, l2_x]
            legs.sort()
            interval = legs[1] - legs[0]
            posture_rew += gaussian(curr_torso, interval / 2 + legs[0], interval / 2) * 0.01

            if lf < 0 or rf < 0:
                done = True
            if curr_torso > 0.5:
                right.append(rf)
                left.append(lf)

            if done:
                break

        dist, max_error = calculate_sin(array_diff(left, right))
        if max_error != 0:
            error = rmse(array_diff(left, right), dist)
            weight = 10.0 / (max_error - objective_rmse)
        else:
            error = 0
            weight = 0

        dist = env.get_body_com("torso")[0]

        actions_std = np.std(action_arr, axis=0)
        action_space = len(action_arr[0])
        variance_rew = 0
        for k in range(action_space):
            # avoid fixed output
            if actions_std[k] >= 0.01:
                variance_rew += 10

        if variance_rew < objective_variance:
            outer_rew = variance_rew
        elif dist < objective_distance:
            outer_rew = objective_variance + dist
        elif weight == 0 or error > objective_rmse:
            outer_rew = objective_distance + variance_rew + weight * (max_error - error)
        elif posture_rew < objective_posture:
            outer_rew = objective_distance + variance_rew + weight * (max_error - objective_rmse) + posture_rew
        else:
            outer_rew = dist + weight * (max_error - error) + variance_rew + posture_rew

        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        return outer_rew

    def multi_objective_trial(self, env, net, steps):
        energy = 0
        observation = env.get_observation()
        action_arr = []
        for i in range(steps):
            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)
            if self.save_simulation:
                action_arr.append(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            state_desc = env.get_state_desc()
            for muscle in sorted(state_desc['muscles'].keys()):
                energy += np.square(state_desc['muscles'][muscle]['activation'])
            if done:
                break
        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)
        return FitnessObj(distance=env.get_state_desc()['body_pos']["pelvis"][0], energy=energy)

    def execute_trial_definitive(self, env, net, steps):
        final_rew = 0
        variables = dict()
        variables['prev_distance'] = max(get_x(env, left=True), get_x(env))
        variables['left_rew'] = 0.0
        variables['right_rew'] = 0.0
        variables['fly_steps'] = 0.0
        observation = env.get_observation()
        for i in range(steps):
            action = net.activate(observation)
            action = self.add_action_for_3d(action)
            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)

            correct = False
            lf_h, rf_h = env.get_state_desc()['body_pos']['toes_l'][1], env.get_state_desc()['body_pos']['toes_r'][1]

            # CORRECT IF 1 FOOT DOWN
            if lf_h <= 0.05 or rf_h <= 0.05:
                correct = True

            current_distance = max(get_x(env, left=True), get_x(env))
            h_pelvis = env.get_state_desc()['body_pos']['pelvis'][1]
            if current_distance > variables['prev_distance']:
                distance = current_distance - variables['prev_distance']
                # distance = variables['info']['forward_reward']
                diff = get_x(env, left=True) - get_x(env)
                if max(get_y(env, left=True), get_y(env)) < h_pelvis:
                    rew = abs(distance * diff)
                    if diff > 0:
                        variables['left_rew'] += rew
                    else:
                        variables['right_rew'] += rew

                variables['prev_distance'] = current_distance

            final_rew += reward
            if done:
                break

        rr = variables['right_rew']
        lr = variables['left_rew']
        final_rew = rr + lr - abs(rr - lr)

        return final_rew

    def eval_genome(self, genome, config):
        env = RewardShapingEnv(visualize=self.visual, seed=1234, difficulty=2, old_input=self.old_input)
        env.change_model(model='2D', difficulty=2, seed=1234)
        if self.reward_function is None:
            self.reward_function = env.energy_consumption_reward
        env.set_reward_function(self.reward_function)
        env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
        if not self.is_a_net:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = genome
        if self.reward_type == 0:
            return self.execute_trial_progressive_reward(env, net, self.steps)
        if self.reward_type == 1:
            return self.execute_trial_with_distance(env, net, self.steps)
        elif self.reward_type == 2:
            return self.execute_trial_with_area(env, net, self.steps)
        elif self.reward_type == 3:
            return self.execute_trial_step_reward(env, net, self.steps)
        elif self.reward_type == 4:
            return self.execute_trial_step_reward(env, net, self.steps)
        elif self.reward_type == 5:
            return self.execute_trial_definitive(env, net, self.steps)
        elif self.reward_type == 6:
            return self.multi_objective_trial(env, net, self.steps)
        elif self.reward_type == 7:
            return self.execute_trial_with_body_in_range_distance(env, net, self.steps)
        else:
            return self.execute_trial(env, net, self.steps)


def print_file(string, file="output.txt"):
    with open(file, "a") as file_out:
        file_out.write(string)


def from_list_to_dict(l):
    d = {}
    for gid, g in l:
        d[gid] = g
    return d


def get_x(env, left=False):
    if left:
        return env.get_state_desc()['body_pos']['tibia_l'][0]
    return env.get_state_desc()['body_pos']['tibia_r'][0]


def get_y(env, left=False):
    if left:
        return env.get_state_desc()['body_pos']['tibia_l'][1]
    return env.get_state_desc()['body_pos']['tibia_r'][1]


def gaussian(x, mu, sig=0.1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))