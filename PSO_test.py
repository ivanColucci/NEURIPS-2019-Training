import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
from osim.env import Arm2DEnv
import pickle
import math
import random
random.seed(42)

def from_arr_to_weights(x, model):
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


if __name__ == "__main__":
    env = Arm2DEnv(visualize=True)
    model = Sequential()
    env.reset(random_target=False)
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(env.get_action_space_size()))
    model.add(Activation('sigmoid'))
    model.compile(Adam(lr=.001, clipnorm=1.), loss=['mae'])

    num_of_weights = 0
    for layer in model.layers:
        tempArr = layer.get_weights()
        if len(tempArr) > 0:
            num_of_weights += tempArr[0].shape[0] * tempArr[0].shape[1] + tempArr[0].shape[1]

    with open("championPSO_100", "rb") as fin:
        champion_x = pickle.load(fin)
    from_arr_to_weights(champion_x, model)
    final_rew = 0
    observation = env.reset(obs_as_dict=False)

    for i in range(1000):
        obs = np.reshape(observation, (1, 1, 339))
        action = model.predict_on_batch(obs)
        observation, reward, done, info = env.step(action[0], obs_as_dict=False)
        # Evaluate fitness based only on reward
        for elem in observation:
            if math.isnan(elem):
                break
        if done:
            break
        final_rew += reward
        print(final_rew)