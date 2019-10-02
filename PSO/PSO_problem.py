import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from osim.env import L2M2019Env
import numpy as np
import random
import math

random.seed(1234)
# environment
env = L2M2019Env(visualize=False, seed=1234, difficulty=2)


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
            currentLayerDim = tempArr[0].shape[0] * tempArr[0].shape[1]
            pesi_layer = x[index:index + currentLayerDim]
            index += currentLayerDim
            bias_layer = x[index:index + tempArr[0].shape[1]]
            index += tempArr[0].shape[1]
            np_x = np.array(pesi_layer)
            np_x = np.reshape(np_x, tempArr[0].shape)
            weights = [np_x, bias_layer]
            layer.set_weights(weights)

#Problem definition
class ball_catching:
    def __init__(self, steps=200, num_of_weights=79382):
        self.steps = steps
        self.num_of_weights = num_of_weights

    def fitness(self, x):
        final_rew = 0

        #imports
        import tensorflow as tf
        from keras import backend as K
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Flatten, Conv1D
        from keras.optimizers import Adam

        # session
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1,
                                allow_soft_placement=True)
        session = tf.compat.v1.Session(config=config)
        K.set_session(session)

        # old model
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(env.get_action_space_size()))
        model.add(Activation('sigmoid'))


        set_model_weights(model, x[0])

        observation = env.reset(obs_as_dict=False, seed=1234)
        for i in range(1, self.steps+1):
            obs = np.reshape(observation, (1, 1, 339))
            action = model.predict_on_batch(obs)
            observation, reward, done, info = env.step(action[0], obs_as_dict=False)
            # Evaluate fitness based only on reward
            for elem in observation:
                if math.isnan(elem):
                    return [0.]
            if done:
                break
            final_rew += reward

        K.clear_session()
        return [-final_rew]

    def fitness_manager(self, xs):
        dimension = len(xs)
        results = []
        for i in range(dimension):
            results.append(self.fitness([xs[i]])[0])
        print(results)
        return results

    def get_bounds(self):
        bounds_up = []
        bounds_down = []
        for i in range(self.num_of_weights):
            bounds_up.append(1)
            bounds_down.append(-1)
        return (bounds_down, bounds_up)