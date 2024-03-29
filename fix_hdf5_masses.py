from math import fabs
from sys import argv

import h5py
import numpy as np

from temperatures import calculate_T
from utils import print_progress

gamma = 5 / 3
YHe = 0.245421
Tcmb0 = 2.7255


def calculate_gas_internal_energy(omegab, hubble_param_, zstart_):
    astart_ = 1.0 / (1.0 + zstart_)
    if fabs(1.0 - gamma) > 1e-7:
        npol = 1.0 / (gamma - 1.0)
    else:
        npol = 1.0
    unitv = 1e5
    adec = 1.0 / (
            160.0 * (omegab * hubble_param_ * hubble_param_ / 0.022) ** (2.0 / 5.0)
    )
    if astart_ < adec:
        Tini = Tcmb0 / astart_
    else:
        Tini = Tcmb0 / astart_ / astart_ * adec
    print("Tini", Tini)
    if Tini > 1.0e4:
        mu = 4.0 / (8.0 - 5.0 * YHe)
    else:
        mu = 4.0 / (1.0 + 3.0 * (1.0 - YHe))
    print("mu", mu)
    ceint_ = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / unitv / unitv
    print("ceint", ceint_)
    return ceint_


def calculate_smoothing_length(boxsize, hubble_param_, levelmax):
    mean_interparte_separation = boxsize / 2 ** levelmax
    print("smoothing length", mean_interparte_separation)
    return mean_interparte_separation


def fix_initial_conditions():
    with h5py.File(argv[1], "r+") as f:
        omegab = f["Cosmology"].attrs["Omega_b"]
        h = f["Cosmology"].attrs["h"]
        zstart = f["Header"].attrs["Redshift"]
        boxsize = f["Header"].attrs["BoxSize"]
        levelmax = f["Header"].attrs["Music_levelmax"]
        internal_energy = calculate_gas_internal_energy(
            omegab=omegab, hubble_param_=h, zstart_=zstart
        )
        smoothing_length = calculate_smoothing_length(
            boxsize=boxsize, hubble_param_=h, levelmax=levelmax
        )
        # exit()
        bary_mass = f["Header"].attrs["MassTable"][0]
        bary_count = f["Header"].attrs["NumPart_Total"][0]
        print("mass table", f["Header"].attrs["MassTable"])
        pt1 = f["PartType0"]
        masses_column = pt1.create_dataset(
            "Masses", data=np.full(bary_count, bary_mass), compression="gzip"
        )
        smoothing_length_column = pt1.create_dataset(
            "SmoothingLength",
            data=np.full(bary_count, smoothing_length),
            compression="gzip",
        )
        internal_energy_column = pt1.create_dataset(
            "InternalEnergy",
            data=np.full(bary_count, internal_energy),
            compression="gzip",
        )




def add_temperature_column():
    with h5py.File(argv[1], "r+") as f:
        pt0 = f["PartType0"]
        us = pt0["InternalEnergies"]
        Ts = []
        i = 0
        total = len(us)
        for u in us:
            print_progress(i, total)
            i += 1
            Ts.append(calculate_T(u))
        pt0.create_dataset(
            "Temperatures",
            data=Ts,
        )


if __name__ == "__main__":
    # fix_initial_conditions()
    add_temperature_column()
