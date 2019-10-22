import numpy as np
from myenv import RewardShapingEnv
import neat
import math

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


def execute_trial(env, net, steps):
    # final_rew = 0
    observation = env.get_observation()
    # Returns the phenotype associated to given genome
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for_3d(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        # final_rew += reward
        if done:
            break
    return env.get_state_desc()['body_pos']['pelvis'][0]


def get_reward(body_y, step_posx):
    dim = len(step_posx)
    for h in body_y:
        if h < 0.8 or h > 0.95:
            return 0
    positions = []
    for i in range(dim):
        if i == 0:
            positions.append(0.4)
        else:
            positions.append(round(positions[-1] + 0.7, 1))
    diff = np.abs(np.subtract(step_posx, positions))
    clip_diff = np.clip(diff, a_min=0.0, a_max=0.7)
    rew = 0.7 - clip_diff
    total_rew = np.sum(rew)
    return total_rew


def execute_trial_step_reward(env, net, steps):
    observation = env.get_observation()
    body_y = []
    step_posx = []
    doing_stepr = False
    doing_stepl = False
    step_threshold = 0.02
    # Returns the phenotype associated to given genome
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for_3d(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        posy_r = env.get_state_desc()["body_pos"]["toes_r"][1]
        posy_l = env.get_state_desc()["body_pos"]["toes_l"][1]
        body_y.append(env.get_state_desc()["body_pos"]["pelvis"][1])

        if posy_r < step_threshold and not doing_stepr:
            doing_stepr = True
            step_posx.append(env.get_state_desc()["body_pos"]["toes_r"][0])
        if posy_l < step_threshold and not doing_stepl:
            doing_stepl = True
            step_posx.append(env.get_state_desc()["body_pos"]["toes_l"][0])

        if doing_stepr and posy_r > step_threshold:
            doing_stepr = False
        if doing_stepl and posy_l > step_threshold:
            doing_stepl = False

        if done:
            break
    return get_reward(body_y, step_posx)


def execute_trial_with_param(env, net, steps):
    final_rew = 0
    observation = env.get_observation()
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for_3d(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        final_rew += reward
        if done:
            break
    return [("fitness", final_rew), ("falcata", 3.2)]


def eval_genome(genome, config, visual=False, is_a_net=False):
    env = RewardShapingEnv(visualize=visual, seed=1234, difficulty=2)
    env.change_model(model='2D', difficulty=2, seed=1234)
    env.set_reward_function(env.distance_reward)
    env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
    if not is_a_net:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        net = genome
    return execute_trial_step_reward(env, net, 1000)

