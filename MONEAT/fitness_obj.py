import math
import numpy as np


def print_file(string, file="output.txt"):
    with open(file, "a") as file_out:
        file_out.write(string)


def to_arrays(vector):
    distance_array = []
    energy_array = []
    for el in vector:
        if type(el) is FitnessObj:
            distance_array.append(el.distance)
            energy_array.append(el.energy)
        else:
            print_file("WRONG TYPE: " + el, file="errors")
    return distance_array, energy_array


def mean_vector(vector):
    d, e = to_arrays(vector)
    return FitnessObj(np.mean(d), np.mean(e))


def std_vector(vector):
    d, e = to_arrays(vector)
    return FitnessObj(np.std(d), np.std(e))


def sum_vector(vector):
    d, e = to_arrays(vector)
    return FitnessObj(np.sum(d), np.sum(e))


MAX_ENERGY = 10000
MAX_DISTANCE = 5


class FitnessObj():

    def __init__(self, distance, energy):
        self.distance = distance
        self.energy_dissipated = energy
        self.energy_remaining = self.take_remaining(energy)

    @staticmethod
    def take_remaining(energy_dissipated):
        return MAX_DISTANCE * (MAX_ENERGY - energy_dissipated) / MAX_ENERGY

    # ************** COMPARATIVE OPERATORS *******************
    def __lt__(self, other):
        if type(other) is float or type(other) is int:
            return self.distance < other
        my_pareto_dist = math.sqrt(math.pow(self.distance, 2) + math.pow(self.energy_remaining, 2))
        other_pareto_dist = math.sqrt(math.pow(other.distance, 2) + math.pow(other.energy_remaining, 2))
        return my_pareto_dist < other_pareto_dist

    def __eq__(self, other):
        if type(other) is float or type(other) is int:
            return self.distance == other
        my_pareto_dist = math.sqrt(math.pow(self.distance, 2) + math.pow(self.energy_remaining, 2))
        other_pareto_dist = math.sqrt(math.pow(other.distance, 2) + math.pow(other.energy_remaining, 2))
        return my_pareto_dist == other_pareto_dist

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    # *************** MATHEMATICAL OPERATORS **********************
    def __add__(self, other):
        if type(other) is float or type(other) is int:
            return FitnessObj(distance=(self.distance + other),
                              energy=self.energy_dissipated)
        return FitnessObj(distance=(self.distance + other.distance),
                          energy=(self.energy_dissipated + other.energy_dissipated))

    def __sub__(self, other):
        if type(other) is float or type(other) is int:
            return FitnessObj(distance=(self.distance - other),
                              energy=self.energy_dissipated)
        return FitnessObj(distance=(self.distance - other.distance),
                          energy=(self.energy_dissipated - other.energy_dissipated))

    def __truediv__(self, other):
        if type(other) is float or type(other) is int:
            return FitnessObj(distance=(self.distance.__truediv__(other)),
                              energy=self.energy_dissipated)
        return FitnessObj(distance=(self.distance.__truediv__(other.distance)),
                          energy=self.energy_dissipated.__truediv__(other.energy_dissipated))

    def __mul__(self, other):
        if type(other) is float or type(other) is int:
            return FitnessObj(distance=(self.distance.__mul__(other)),
                              energy=self.energy_dissipated)
        return FitnessObj(distance=(self.distance.__mul__(other.distance)),
                          energy=self.energy_dissipated.__mul__(other.energy_dissipated))

    # ************** OTHER OPERATORS ********************
    def __str__(self):
        return 'D: {} E: {}'.format(str(round(self.distance, 2)), str(round(self.energy_dissipated, 2)))

    def __copy__(self):
        return FitnessObj(self.distance, self.energy_dissipated)

    def __format__(self, format_spec):
        return self.__str__()

    def __round__(self, n=None):
        return FitnessObj(round(self.distance, n), round(self.energy_dissipated, n))
