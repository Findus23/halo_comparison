import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def V(r):
    return 4 * np.pi * r ** 3 / 3


def cumulative_mass_profile(particles_in_halos: pd.DataFrame, halo: pd.Series):
    print(type(particles_in_halos))
    centre = np.array([halo.X, halo.Y, halo.Z])
    positions = particles_in_halos[["X", "Y", "Z"]].to_numpy()
    print(positions)
    print(positions.shape)
    distances = np.linalg.norm(positions - centre, axis=1)
    group_radius = distances.max()
    normalized_distances = distances / group_radius

    num_bins = 30
    log_radial_bins = np.geomspace(0.01, 2, num_bins)

    bin_counts = []
    bin_densities = []
    for k in range(num_bins - 1):
        bin_start = log_radial_bins[k]
        bin_end = log_radial_bins[k + 1]
        in_bin = np.where((bin_start < normalized_distances) & (normalized_distances < bin_end))[0]
        count = in_bin.shape[0]
        volume = V(bin_end * group_radius) - V(bin_start * group_radius)
        bin_counts.append(count)
        density = count / volume
        bin_densities.append(density)
    print(bin_counts)
    print(bin_densities)

    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    ax2 = ax.twinx()

    ax.loglog(log_radial_bins[:-1], bin_counts, label="counts")
    ax2.loglog(log_radial_bins[:-1], bin_densities, label="densities",c="C1")
    ax.set_xlabel(r'R / R$_\mathrm{group}$')
    ax.set_ylabel(r'M [$10^{10} \mathrm{M}_\odot$]')
    ax2.set_ylabel("density")

    plt.show()

    return bin_counts, bin_densities
