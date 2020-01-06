import os
import pickle
import matplotlib.pyplot as plt
import neat
import numpy as np
from myenv import RewardShapingEnv

from NEAT.utils.rmse import calculate_sin, array_diff

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


class Evaluator:
    def __init__(self, save_simulation=False, load_simulation=False, plot=False,
                 file_to_load="actions", visual=False, is_a_net=False, steps=1000, video=False, my_env=False,
                 done=True, seed=1234):
        self.visual = visual
        self.is_a_net = is_a_net
        self.steps = steps
        self.load_simulation = load_simulation
        self.save_simulation = save_simulation
        self.file_name = file_to_load
        self.make_video = video
        self.video = None
        self.my_env = my_env
        self.plot = False
        self.plot_rmse = plot
        self.done = done
        self.seed = seed
        if self.load_simulation:
            local_dir = os.path.dirname(__file__)
            with open(os.path.join(local_dir, self.file_name), 'rb') as f:
                self.action_arr = pickle.load(f)

    @staticmethod
    def observation_reduction(observation):
        return observation[:97]

    def get_plot(self):
        self.plot = True

    def load_next_action(self, i):
        return self.action_arr[i]

    @staticmethod
    def add_action_for_3d(action):
        Fmax_ABD = 4460.290481
        Fmax_ADD = 3931.8
        r_leg, l_leg = action[:9], action[9:]
        full_action = []
        full_action.append(0.1)
        full_action.append(0.1 * Fmax_ADD / Fmax_ABD)
        for el in r_leg:
            full_action.append(el)
        full_action.append(0.1)
        full_action.append(0.1 * Fmax_ADD / Fmax_ABD)
        for el in l_leg:
            full_action.append(el)
        return full_action

    def execute_trial(self, env, net, steps):
        # Env reset
        observation = env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
        # init variables
        action_arr = []
        phenotype = []
        speed_arr = []
        left = []
        right = []
        torso_y = []
        fall = 0
        variables = dict()
        distance = 0.0

        # simulation
        for i in range(steps):

            if not self.load_simulation:
                action = net.activate(observation)
                action = self.add_action_for_3d(action)
            else:
                action = self.load_next_action(i)

            action_arr.append(action)

            if self.visual:
                env.render()

            observation, reward, done, info = env.step(action, project=True, obs_as_dict=False)
            t = env.osim_model.stepsize
            curr_dist = env.get_state_desc()['body_pos']['pelvis'][0]
            diff_dist = curr_dist - distance
            distance = curr_dist
            curr_speed = diff_dist/t
            speed_arr.append(curr_speed)

            torso_y.append(env.get_state_desc()['body_pos']['pelvis'][1])
            left.append(get_x(env, left=True))
            right.append(get_x(env))

            if self.done and done:
                fall = 1
                break

        distance = env.get_state_desc()['body_pos']['pelvis'][0]

        diff_arr = array_diff(left, right)
        info = calculate_sin(diff_arr, params_only=True)

        outer_rew = hook_outer(variables)

        mean_speed = np.mean(speed_arr)

        phenotype.append(np.clip(distance / 10, 0, 1))
        phenotype.append(np.clip(mean_speed / 5, 0, 1))
        phenotype.append(np.clip(info['value'], 0, 1))
        phenotype.append(fall)

        if self.save_simulation:
            with open(self.file_name, 'wb') as f:
                pickle.dump(action_arr, f)

        return outer_rew, phenotype, action_arr

    def eval_genome(self, genome, config):
        env = RewardShapingEnv(visualize=self.visual, seed=1234, difficulty=2)
        env.change_model(model='2D', difficulty=2, seed=1234)
        env.set_reward_function(env.energy_consumption_reward)
        env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)

        if not self.is_a_net:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = genome

        fitness, phenotype, action_arr = self.execute_trial(env, net, self.steps)

        if self.plot:
            plot_actions(action_arr)
        return fitness, phenotype


def print_file(string, file="output.txt"):
    with open(file, "a") as file_out:
        file_out.write(string)


def from_list_to_dict(l):
    d = {}
    for gid, g in l:
        d[gid] = g
    return d


def plot_actions(action_arr):
    np_actions_arr = np.array(action_arr)

    text = ["Coscia", "Gamba", "Piede"]
    fig1, axs1 = plt.subplots(3)
    fig2, axs2 = plt.subplots(3)
    fig1.suptitle('Azioni nel tempo parte destra')
    fig2.suptitle('Azioni nel tempo parte sinistra')
    for i in range(3):
        act_i = np_actions_arr[:, i]
        axs1[i].set_title(text[i])
        axs1[i].plot(act_i)
        act_i_shift = np_actions_arr[:, i + 3]
        axs2[i].set_title(text[i])
        axs2[i].plot(act_i_shift)
    plt.show()


def get_shift(arr, step):
    return arr[step:] + arr[0:step]


def get_smaller_diff(arr_1, arr_2):
    ac = np.array(arr_1)
    best_diff = 2000

    for step in range(len(arr_1)):
        tr = get_shift(arr_2, step)
        difference = sum(abs(ac - np.array(tr)))
        if difference == 0:
            return 0
        if difference < best_diff:
            best_diff = difference

    return best_diff


def get_x(env, left=False):
    if left:
        return env.get_state_desc()['body_pos']['tibia_l'][0]
    return env.get_state_desc()['body_pos']['tibia_r'][0]


def get_y(env, left=False):
    if left:
        return env.get_state_desc()['body_pos']['tibia_l'][1]
    return env.get_state_desc()['body_pos']['tibia_r'][1]


def hook_inner(env, variables):
    if 'prev_distance' not in variables:
        variables['prev_distance'] = max(get_x(env, left=True), get_x(env))
        variables['left_rew'] = 0.0
        variables['right_rew'] = 0.0
        variables['fly_steps'] = 0.0

    h_tol = 0.3441
    correct = False
    lf_h, rf_h = env.get_state_desc()['body_pos']['toes_l'][1], env.get_state_desc()['body_pos']['toes_r'][1]

    # CORRECT IF 1 FOOT DOWN
    if lf_h <= 0.05 or rf_h <= 0.05:
        correct = True

    current_distance = max(get_x(env, left=True), get_x(env))
    h_torso = env.get_state_desc()['body_pos']['pelvis'][1]
    if current_distance > variables['prev_distance']:
        distance = current_distance - variables['prev_distance']
        diff = get_x(env, left=True) - get_x(env)
        # PENALTY h(LEG) > h(TORSO)
        if max(get_y(env, left=True), get_y(env)) < h_torso:
            rew = abs(distance * diff)
            if diff > 0:
                variables['left_rew'] += rew
            else:
                variables['right_rew'] += rew

        variables['prev_distance'] = current_distance

    return 0.0


def hook_outer(variables):
    rr = variables['right_rew']
    lr = variables['left_rew']
    leg_rew = rr + lr - abs(rr - lr)
    return leg_rew
