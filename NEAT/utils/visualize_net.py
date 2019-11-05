from NEAT.utils.visualize import draw_net
import neat
import os
import pickle
from NEAT.DefaultTournament.my_reproduction import TournamentReproduction


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-osim')
rep_type = 2

if rep_type == 2:
    rep_class = TournamentReproduction
else:
    rep_class = neat.DefaultReproduction

config = neat.Config(neat.DefaultGenome, rep_class,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

load_from_checkpoint = False

with open('winner_genome_tournament', 'rb') as f:
    winner = pickle.load(f)

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

# Show output of the most fit genome against training data.
draw_net(config, winner, True)