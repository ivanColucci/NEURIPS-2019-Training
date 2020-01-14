from neat.genome import DefaultGenome, DefaultNodeGene, DefaultConnectionGene
from MONEAT.mogenome import MyDefaultGenomeConfig


class MyGenome(DefaultGenome):

    def __init__(self, key):
        super().__init__(key)
        self.phenotype = None

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return MyDefaultGenomeConfig(param_dict)
