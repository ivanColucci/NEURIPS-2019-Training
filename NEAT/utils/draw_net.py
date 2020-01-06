from graphviz import Digraph
import neat
import pickle
from NEAT.DefaultTournament.my_genome import MyGenome
from NEAT.EliteTournament.elite_reproduction import EliteReproduction


def divide_net_layers (genome, config):
    layers = dict()
    layers['input'] = set()
    for k in config.genome_config.input_keys:
        layers['input'].add(k)

    layers['output'] = set()
    for k in config.genome_config.output_keys:
        layers['output'].add(k)

    connections = []
    layers['hidden1'] = set()
    layers['connections'] = set()
    index = 0
    for cg in genome.connections.values():
        input_key, output_key = cg.key
        layers['connections'].add(cg.key)
        #print(cg.key)
        if layers['input'].__contains__(input_key) and not layers['output'].__contains__(output_key):
            layers['hidden1'].add(output_key)
        elif layers['output'].__contains__(output_key):
            continue
        else:
            connections.append(cg)
    if (len(layers['hidden1'])) > 0:
        index += 1

    while len(connections) > 0:
        previous_layer = 'hidden' + str(index)
        name_layer = 'hidden' + str(index+1)
        layers[name_layer] = set()

        for _ in range(len(connections)):
            cg = connections.pop(0)
            input_key, output_key = cg.key
            if layers[previous_layer].__contains__(input_key) and not layers['output'].__contains__(output_key):
                layers[name_layer].add(output_key)
            elif layers['output'].__contains__(output_key):
                continue
            else:
                connections.append(cg)
        index += 1
    layers['n_layer'] = index
    print(layers['input'])
    for i in range(1,1+index):
        name_layer = 'hidden' + str(i)
        print(layers[name_layer])
    print(layers['output'])

    return layers


def plot_layers(genome, config, filename):
    layers = divide_net_layers(genome, config)
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'style': 'filled',
        'width': '0.2'}
    g = Digraph(format='png', node_attr=node_attrs)
    g.graph_attr.update(splines="false", nodesep='1', ranksep='2');
    with g.subgraph(name="input") as c:
        for n in layers['input']:
            c.attr(rank='same')
            c.node(str(n), str(n), color="#2ecc71")

    for i in range(1, 1+layers['n_layer']):
        with g.subgraph(name="hidden"+str(i)) as c:
            name_layer = 'hidden' + str(i)
            for n in layers[name_layer]:
                c.attr(rank='same')
                c.node(str(n), str(n), color="#3498db")

    with g.subgraph(name="output") as c:
        for n in layers['output']:
            c.attr(rank='same')
            c.node(str(n), str(n), color="red")

    for edge in layers['connections']:
        g.edge(str(edge[0]), str(edge[1]), arrowhead="none")

    g.render(filename, view=True)

with open('../../winner_genome','rb') as f:
    winner = pickle.load(f)
    config_file = "../../config"
    config = neat.Config(MyGenome, EliteReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    plot_layers(winner, config, 'prova')

