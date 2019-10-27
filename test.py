from osim.env import L2M2019Env
import numpy as np
import neat
import pickle

from NEAT.my_reproduction import TournamentReproduction

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


env = L2M2019Env(visualize=False, seed=1234, difficulty=2)
env.change_model(model='2D', difficulty=2, seed=1234)
observation = env.reset(project=True, seed=1234, obs_as_dict=False, init_pose=INIT_POSE)
values = observation[-97:]

print("\n\n**************************** OSIM FA SCHFIO ****************************\n")
h = 1234
for value in values:
    h = hash(h+value)
print("hash code (di python) della prima osservazione:", h)
env.action_space.seed(1234)
fixed_action = []
for i in range(22):
    fixed_action.append(0.0)
tr = 0
print("Azione fissa:", fixed_action)

for i in range(500):
    observation, reward, done, info = env.step(fixed_action, project=True, obs_as_dict=False)
    values = observation[-97:]
    for value in values:
        h = hash(h + value)
    tr += reward
    if done:
        break
print("reward dell'ambiente:", tr)
print("hash code (di python) della concatenazione delle varie osservazioni:", h)
