from itertools import product

import numpy as np


class IDScaler:
    def __init__(self, Nres_min: int, Nres_max: int):
        assert Nres_max % Nres_min == 0
        self.Nres_min = Nres_min
        self.Nres_max = Nres_max
        self.N = Nres_max // Nres_min
        self.shifts = []
        for shift in product(range(self.N), repeat=3):
            self.shifts.append(np.array(shift))

    @staticmethod
    def original_position(Nres: int, particle_ID: int):
        particle_k = particle_ID % Nres
        particle_j = ((particle_ID - particle_k) // Nres) % Nres
        particle_i = ((particle_ID - particle_k) // Nres - particle_j) // Nres
        return np.array([particle_i, particle_j, particle_k])

    def upscale(self, particle_ID: int):
        orig = self.original_position(self.Nres_min, particle_ID)
        mult = orig * self.N
        for shift in self.shifts:
            variant = mult + shift
            yield ((variant[0] * self.Nres_max) + variant[1]) * self.Nres_max + variant[2]

    def downscale(self, particle_ID: int):
        orig = self.original_position(self.Nres_max, particle_ID)
        mult = np.floor_divide(orig, self.N)
        return ((mult[0] * self.Nres_min) + mult[1]) * self.Nres_min + mult[2]


if __name__ == "__main__":
    test_particle = np.array([0, 0, 127])
    # maximum_test = np.array([127, 127, 127]) #this works, Nres - 1 is the maximum for (i,j,k)

    Nres_1 = 128
    Nres_2 = 256

    test_particle_id = ((test_particle[0] * Nres_1) + test_particle[1]) * Nres_1 + test_particle[2]
    print(test_particle_id)

    scaler = IDScaler(Nres_1, Nres_2)

    particle_ID_1_converted = scaler.upscale(test_particle_id)

    for id in particle_ID_1_converted:
        reverse = scaler.downscale(id)
        print(id, reverse)
        assert reverse == test_particle_id
