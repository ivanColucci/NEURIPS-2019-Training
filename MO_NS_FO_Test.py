import os
import pickle

import neat

from MO_NS_FO.MO_NS_FO_EliteReproduction import EliteReproduction
from MO_NS_FO.MO_NS_FO_Evaluator import Evaluator
from MO_NS_FO.MO_NS_FO_Genome import Genome
from MO_NS_FO.MO_NS_FO_Reproduction import Reproduction
from MO_NS_FO.MO_NS_FO_Species import SpeciesSet
from MO_NS_FO.MO_NS_FO_Stagnation import Stagnation
from NEAT.utils.utilities import step_activation, reverse_activation


def run(visual_env=True, visual_video=False, elite=True, human=True):
    i = 1
    local_dir = os.path.dirname(__file__)
    with open("front_0", "rb") as f:
        front = pickle.load(f)
        if human:
            model = 'Humanoid-v3'
            if elite:
                config_path = os.path.join(local_dir, 'MO_NS_FO_EliteHumanoidConfig')
            else:
                config_path = os.path.join(local_dir, 'MO_NS_FO_HumanoidConfig0')
        else:
            model = 'Walker2d-v3'
            if elite:
                config_path = os.path.join(local_dir, 'MO_NS_FO_EliteConfig')
            else:
                config_path = os.path.join(local_dir, 'MO_NS_FO_Config0')

        if elite:
            reproduction = EliteReproduction
        else:
            reproduction = Reproduction

        config = neat.Config(Genome, reproduction,
                             SpeciesSet, Stagnation,
                             config_path)

        config.genome_config.add_activation('step', step_activation)
        config.genome_config.add_activation('reverse', reverse_activation)
        evaluator = Evaluator(visual=visual_env, model_name=model, video=visual_video, my_env=False, plot=False,
                              steps=5000, done=True, seed=1234)

        front.sort(key=lambda x: x[1].fitness)
        for _, winner in front:
            print("Individuo n°", i)
            if i in [len(front)]:
                phenotype = evaluator.eval_genome(winner, config)
                print(phenotype)
            print('Fitness: {}'.format(winner.fitness))
            print('Size: {}'.format(winner.size()))
            print("\n")
            # input()
            i += 1


run(visual_env=True, visual_video=False, elite=True, human=True)
