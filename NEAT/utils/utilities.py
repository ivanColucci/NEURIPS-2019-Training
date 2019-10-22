import numpy as np
from myenv import RewardShapingEnv
import neat

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
    final_rew = 0
    observation = env.get_observation()
    # Returns the phenotype associated to given genome
    for i in range(steps):
        action = net.activate(observation)
        action = add_action_for_3d(action)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=False)
        #print(reward)
        final_rew += reward
        if done:
            break
        #print("Final_rew: {}; Distance: {}".format(final_rew, env.get_state_desc()['body_pos']['pelvis'][0]))
    return final_rew + 1000*env.get_state_desc()['body_pos']['pelvis'][0]


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
    env.set_reward_function(env.pelvis_height)
    env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
    if not is_a_net:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        net = genome
    return execute_trial(env, net, 1000)

