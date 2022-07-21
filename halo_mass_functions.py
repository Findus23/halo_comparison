from math import log
from pathlib import Path

import numpy as np
# from colossus.cosmology import cosmology
# from colossus.lss import mass_function
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paths import base_dir, has_1024_simulations
from read_vr_files import read_velo_halos
from utils import print_progress


def counts_without_inf(number_halos):
    with np.errstate(divide='ignore', invalid='ignore'):
        number_halos_inverse = 1 / np.sqrt(number_halos)
        number_halos_inverse[np.abs(number_halos_inverse) == np.inf] = 0

    return number_halos_inverse


def monofonic_tests():
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    resolutions = [128, 256, 512]
    if has_1024_simulations:
        resolutions.append(1024)

    for i, waveform in enumerate(["DB2", "shannon"]):
        for j, resolution in enumerate([128, 256, 512]):
            print(waveform, resolution)
            dir = base_dir / f"{waveform}_{resolution}_100"
            halos = read_velo_halos(dir)

            # halos.to_csv("weird_halos.csv")
            halo_masses: np.ndarray = halos["Mvir"].to_numpy()

            Ns, deltas, left_edges, number_densities, lower_error_limit, upper_error_limit = halo_mass_function(
                halo_masses)

            ax.set_xscale("log")
            ax.set_yscale("log")

            # ax.bar(centers, number_densities, width=widths, log=True, fill=False)
            name = f"{waveform} {resolution}"
            ax.step(left_edges, number_densities, where="post", color=f"C{i}", linestyle=linestyles[j], label=name)

            ax.fill_between(
                left_edges,
                lower_error_limit,
                upper_error_limit, alpha=.5, linewidth=0, step='post')

            # break
        # break
    plt.legend()
    fig.savefig(Path(f"~/tmp/halo_mass_function.pdf").expanduser())
    plt.show()


def halo_mass_function(halo_masses, num_bins=30, sim_volume=100 ** 3):
    bins = np.geomspace(halo_masses.min(), halo_masses.max(), num_bins + 1)
    digits = np.digitize(halo_masses, bins)
    number_densities = []
    widths = []
    centers = []
    left_edges = []
    Ns = []
    deltas = []
    for bin_id in range(num_bins):
        print_progress(bin_id, num_bins)
        mass_low = bins[bin_id]
        mass_high = bins[bin_id + 1]
        counter = 0
        for val in halo_masses:
            if mass_low <= val < mass_high:
                counter += 1
        delta_mass = mass_high - mass_low
        delta_log_mass = log(mass_high) - log(mass_low)
        widths.append(delta_mass)
        centers.append(mass_low + delta_mass / 2)
        left_edges.append(mass_low)

        values = np.where(digits == bin_id + 1)[0]
        # print(halo_masses[values])
        # print(values)
        num_halos = values.shape[0]
        assert num_halos == counter
        nd = num_halos / sim_volume / delta_log_mass
        number_densities.append(nd)
        Ns.append(num_halos)
        deltas.append(delta_mass)
    deltas = np.array(deltas)
    Ns = np.array(Ns)
    left_edges = np.array(left_edges)
    number_densities = np.array(number_densities)
    lower_error_limit = number_densities - counts_without_inf(Ns) / sim_volume / deltas
    upper_error_limit = number_densities + counts_without_inf(Ns) / sim_volume / deltas

    return Ns, deltas, left_edges, number_densities, lower_error_limit, upper_error_limit


def hmf_from_rockstar_tree(file: Path):
    masses = []
    with file.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split()
            Mvir = float(cols[10])
            masses.append(Mvir)
    masses = np.array(masses)
    # agora_box_h = 0.702
    # masses /= agora_box_h
    box_size = 85.47
    Ns, deltas, left_edges, number_densities, lower_error_limit, upper_error_limit = halo_mass_function(
        masses,
        num_bins=50,
        sim_volume=box_size ** 3
    )
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Halo Mass [$M_\\odot$]")
    ax.set_ylabel("Number Density [$\\textrm{\\#}/Mpc^3/dlogM$]")
    ax.step(left_edges, number_densities, where="post")
    plank_cosmo = cosmology.cosmologies['planck18']
    auriga_cosmo = {
        "sigma8": 0.807,
        "H0": 70.2,
        "Om0": 0.272,
        "Ob0": 0.0455,
        "ns": 0.961
    }
    cosmology.addCosmology('aurigaCosmo', params={**plank_cosmo, **auriga_cosmo})
    cosmology.setCosmology('aurigaCosmo')
    print(cosmology.getCurrent())
    mfunc = mass_function.massFunction(left_edges, 1, mdef='vir', model='tinker08', q_out='dndlnM')

    ax.plot(left_edges, mfunc)

    ax.fill_between(
        left_edges,
        lower_error_limit,
        upper_error_limit, alpha=.5, linewidth=0, step='post')

    plt.show()


if __name__ == '__main__':
    monofonic_tests()
    # hmf_from_rockstar_tree(Path(argv[1]))
