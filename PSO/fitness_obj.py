import math

class FitnessObj():

    def __init__(self, distance, energy):
        self.distance = 20 - distance
        self.energy = energy

    def __lt__(self, other):
        my_pareto_dist = math.sqrt(math.pow(self.distance, 2) + math.pow(self.energy, 2))
        other_pareto_dist = math.sqrt(math.pow(other.distance, 2) + math.pow(other.energy, 2))
        return my_pareto_dist < other_pareto_dist

    def __eq__(self, other):
        my_pareto_dist = math.sqrt(math.pow(self.distance, 2) + math.pow(self.energy, 2))
        other_pareto_dist = math.sqrt(math.pow(other.distance, 2) + math.pow(other.energy, 2))
        return my_pareto_dist == other_pareto_dist

    def __le__(self, other):
        return self == other or self < other

    def __str__(self):
        return 'Distanza: ' + str(self.distance) + ' Energia: ' + str(self.energy)

    def __copy__(self):
        return FitnessObj(20-self.distance, self.energy)