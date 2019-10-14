from neat.genome import DefaultGenome
import math


class MyGenome(DefaultGenome):

    def __init__(self, params):
        super().__init__(params)
        self.param = {}

    def distance(self, other, config):
        distance = 0.0
        for k, v in self.param.items():
            distance += abs(v - other.param[k])
        return distance
