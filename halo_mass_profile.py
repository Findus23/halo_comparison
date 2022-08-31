import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from readfiles import ParticlesMeta, read_file, read_halo_file
from temperatures import calculate_T
from utils import print_progress


def V(r):
    return 4 / 3 * np.pi * r ** 3


def halo_mass_profile(
        positions: np.ndarray,
        center: np.ndarray,
        particles_meta: ParticlesMeta,
        rmin: float,
        rmax: float,
        plot=False,
        num_bins=30,
):
    distances = np.linalg.norm(positions - center, axis=1)

    log_radial_bins = np.geomspace(rmin, rmax, num_bins)

    bin_masses = []
    bin_densities = []
    for k in range(num_bins - 1):
        bin_start = log_radial_bins[k]
        bin_end = log_radial_bins[k + 1]
        in_bin = np.where((bin_start < distances) & (distances < bin_end))[0]
        count = in_bin.shape[0]
        mass = count * particles_meta.particle_mass
        volume = V(bin_end) - V(bin_start)
        bin_masses.append(mass)
        density = mass / volume
        bin_densities.append(density)

    bin_masses = np.array(bin_masses)
    bin_densities = np.array(bin_densities)
    bin_masses = np.cumsum(bin_masses)

    if plot:
        fig: Figure = plt.figure()
        ax: Axes = fig.gca()

        ax2 = ax.twinx()

        ax.loglog(log_radial_bins[:-1], bin_masses, label="counts")
        ax2.loglog(log_radial_bins[:-1], bin_densities, label="densities", c="C1")
        # ax.set_xlabel(r'R / R$_\mathrm{group}$')
        ax.set_ylabel(r"M [$10^{10} \mathrm{M}_\odot$]")
        ax2.set_ylabel("density [$\\frac{10^{10} \\mathrm{M}_\\odot}{Mpc^3}$]")
        plt.legend()
        plt.show()

    return log_radial_bins, bin_masses, bin_densities, center


def property_profile(positions: np.ndarray, center: np.ndarray, masses: np.ndarray, properties: Dict[str, np.ndarray],
                     rmin: float, rmax: float, num_bins: int):
    distances = np.linalg.norm(positions - center, axis=1)
    log_radial_bins = np.geomspace(rmin, rmax, num_bins)

    means = {}
    for key in properties.keys():
        means[key] = []
    for k in range(num_bins - 1):
        bin_start = log_radial_bins[k]
        bin_end = log_radial_bins[k + 1]
        print_progress(k, num_bins - 2, bin_end)
        in_bin = np.where((bin_start < distances) & (distances < bin_end))[0]
        masses_in_ring = masses[in_bin]
        for property, property_values in properties.items():
            if property == "InternalEnergies":
                continue
            prop_in_ring = property_values[in_bin]
            if property == "Temperatures":
                prop_in_ring = np.array([calculate_T(u) for u in prop_in_ring])

            # mean_in_ring_unweighted = np.mean(prop_in_ring)
            mean_in_ring = (prop_in_ring * masses_in_ring).sum() / masses_in_ring.sum()
            # print(mean_in_ring_unweighted, mean_in_ring)
            means[property].append(mean_in_ring)

    return log_radial_bins, means


if __name__ == "__main__":
    input_file = Path(sys.argv[1])
    df, particles_meta = read_file(input_file)
    df_halos = read_halo_file(input_file.with_name("fof_" + input_file.name))

    print(df)
    halo_id = 1
    while True:
        particles_in_halo = df.loc[df["FOFGroupIDs"] == halo_id]
        if len(particles_in_halo) > 1:
            break
        halo_id += 1

    halo = df_halos.loc[halo_id]

    halo_mass_profile(particles_in_halo[["X", "Y", "Z"]].to_numpy(), halo, particles_meta, plot=True)
