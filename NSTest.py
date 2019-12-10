import pickle
from NS.NSReproduction import NSReproduction
from NS.NSEvaluator import NSEvaluator
from NS.NSStagnation import NSStagnation
from NS.NSSpecies import NSSpeciesSet
from NS.NSGenome import NSGenome
from NEAT.utils.utilities import step_activation, reverse_activation
import neat


def run(visual_env=True, visual_video=False, human=False):
    i = 1
    with open("archive_checkpoint_0", "rb") as f:
        archive = pickle.load(f)
        config_file = 'NSConfig0'
        model = 'Walker2d-v3'
        if human:
            config_file = "NSHumanoidConfig0"
            model = "Humanoid-v3"

        config = neat.Config(NSGenome, NSReproduction,
                         NSSpeciesSet, NSStagnation,
                         config_file)
        config.genome_config.add_activation('step', step_activation)
        config.genome_config.add_activation('reverse', reverse_activation)
        evaluator = NSEvaluator(visual=visual_env, model_name=model, video=visual_video, my_env=False, plot=False,
                                steps=5000, done=True, seed=1234)

        archive.sort(key=lambda x: x.fitness[0])
        for winner in archive:
            print("Individuo n°", i)
            if i in [len(archive)]:
                phenotype = evaluator.eval_genome(winner, config)
                print(phenotype)
            print('Fitness: {}'.format(winner.fitness))
            print('Size: {}'.format(winner.size()))
            print("\n")
            # input()
            i += 1


run(visual_env=True, visual_video=False, human=True)
