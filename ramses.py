from pathlib import Path

import numpy as np
import pynbody
from pynbody.array import SimArray
from pynbody.snapshot.ramses import RamsesSnap

from readfiles import ParticlesMeta


def load_ramses_data(ramses_dir: Path):
    s: RamsesSnap = pynbody.load(str(ramses_dir))
    mass_array: SimArray = s.dm["mass"]
    coord_array: SimArray = s.dm["pos"]
    a = s.properties["a"]
    print("RAMSES a", a)

    masses = np.asarray(mass_array.in_units("1e10 Msol"))
    high_res_mass = np.amin(np.unique(masses))  # get lowest mass of particles
    is_high_res_particle = masses == high_res_mass

    coordinates = np.asarray(coord_array.in_units("Mpc"))
    hr_coordinates = coordinates[is_high_res_particle] / a

    particles_meta = ParticlesMeta(particle_mass=high_res_mass)
    center = np.median(hr_coordinates, axis=0)
    return hr_coordinates, particles_meta, center
