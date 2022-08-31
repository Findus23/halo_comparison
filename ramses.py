from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pynbody
from pynbody.analysis.profile import Profile
from pynbody.array import SimArray
from pynbody.snapshot.ramses import RamsesSnap

from readfiles import ParticlesMeta
from utils import create_figure


def load_ramses_data(ramses_dir: Path):
    s: RamsesSnap = pynbody.load(str(ramses_dir))
    mass_array: SimArray = s.dm["mass"]
    coord_array: SimArray = s.dm["pos"]
    a = s.properties["a"]
    print("RAMSES a", a)
    # p = Profile(s.gas, ndim=3)
    # s.gas["pos"]-=
    # fig,ax=create_figure()
    # ax.plot(p['rbins'], p['density'], 'k')
    # plt.show()
    # exit()
    masses = np.asarray(mass_array.in_units("1e10 Msol"))
    high_res_mass = np.amin(np.unique(masses))  # get lowest mass of particles
    is_high_res_particle = masses == high_res_mass

    coordinates = np.asarray(coord_array.in_units("Mpc"))
    hr_coordinates = coordinates[is_high_res_particle] / a

    particles_meta = ParticlesMeta(particle_mass=high_res_mass)
    center = np.median(hr_coordinates, axis=0)

    return hr_coordinates, particles_meta, center


def get_slice_argument(extent: List[float], center: List[float], ramses_dir: Path, depth: float):
    xmin, xmax, ymin, ymax = extent
    _, _, zcenter = center
    arguments = {
        "x": (xmin + xmax) / 2,
        "y": (ymin + ymax) / 2,
        "z": zcenter,
        "w": xmax - xmin,
        "h": ymax - ymin,
        "d": depth,
        "l": 12
    }
    from paths import ramses_imager
    args = [str(ramses_imager)]
    for k, v in arguments.items():
        args.append(f"-{k} {v}")

    args.append(str(ramses_dir / "info_00009.txt"))
    return args, ramses_imager.parent


def load_slice_data(file: Path):
    with file.open("rb") as infile:
        np.fromfile(file=infile, dtype=np.int32, count=1)
        [nx, ny] = np.fromfile(file=infile, dtype=np.int32, count=2)
        np.fromfile(file=infile, dtype=np.int32, count=1)

        np.fromfile(file=infile, dtype=np.int32, count=1)
        data: np.ndarray = np.fromfile(file=infile, dtype=np.float32, count=nx * ny)
        np.fromfile(file=infile, dtype=np.int32, count=1)
        print("NEGATIVE", (data < 0).sum())
        # np.fromfile(file=infile, dtype=np.int32, count=1)
        # cm_per_px = np.fromfile(file=infile, dtype=np.float64, count=1)[0]
        # np.fromfile(file=infile, dtype=np.int32, count=1)
    return data.reshape((nx, ny))
