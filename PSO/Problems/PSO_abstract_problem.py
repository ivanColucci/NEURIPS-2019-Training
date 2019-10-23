import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from myenv import RewardShapingEnv
import numpy as np
import random
import math
from PSO.fitness_obj import FitnessObj

random.seed(1234)

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
    -1.745329251994329478e-01]) # ankle flex

#Returns the number of the weights of the given model
def count_weights(model):
    # dimensione arr di pesi
    num_of_weights = 0
    for layer in model.layers:
        tempArr = layer.get_weights()
        if len(tempArr) > 0:
            num_of_weights += tempArr[0].shape[0] * tempArr[0].shape[1] + tempArr[0].shape[1]
    return num_of_weights

#Sets the weights of the given model to x
def set_model_weights(model, x):
    index = 0
    for layer in model.layers:
        tempArr = layer.get_weights()
        if len(tempArr) > 0:
            currentLayerDim = 1
            for shape in tempArr[0].shape:
                currentLayerDim *= shape
            pesi_layer = x[index:index + currentLayerDim]
            index += currentLayerDim
            bias_layer = x[index:index + tempArr[0].shape[-1]]
            index += tempArr[0].shape[1]
            np_x = np.array(pesi_layer)
            np_x = np.reshape(np_x, tempArr[0].shape)
            weights = [np_x, bias_layer]
            layer.set_weights(weights)

#Problem definition
class WalkingProblem:
    def __init__(self, steps=1000, num_of_weights=289222):
        self.steps = steps
        self.num_of_weights = num_of_weights

    def get_energy(self, env):
        state_desc = env.get_state_desc()
        ACT2 = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
        return ACT2

    def fitness_hook(self, model, env):
        pass

    def reward_hook(self, env):
        pass

    def fitness(self, x):
        #imports
        import tensorflow as tf
        from keras import backend as K
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

        env = RewardShapingEnv(visualize=False, seed=1234, difficulty=2)
        # env.set_reward_function(env.distance_reward)
        env = self.reward_hook(env)
        env.change_model(model='2D', difficulty=2, seed=1234)
        env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)

        # session
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1,
                                allow_soft_placement=True)
        session = tf.compat.v1.Session(config=config)
        K.set_session(session)

        # new model
        model = Sequential()
        model.add(Flatten(input_shape=(339, 1)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(env.get_action_space_size(), activation='sigmoid'))
        # model.summary()
        set_model_weights(model, x[0])

        result_object = self.fitness_hook(model, env)
        K.clear_session()
        return result_object

    def fitness_manager(self, xs):
        dimension = len(xs)
        results = []
        for i in range(dimension):
            results.append(self.fitness([xs[i]]))
        return results

    def get_bounds(self):
        bounds_up = []
        bounds_down = []
        for i in range(self.num_of_weights):
            bounds_up.append(30)
            bounds_down.append(-30)
        return (bounds_down, bounds_up)