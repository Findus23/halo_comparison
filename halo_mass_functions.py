from math import log
from pathlib import Path

import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paths import base_dir, has_1024_simulations
from read_vr_files import read_velo_halos
from readfiles import read_gadget_halos
from utils import print_progress, figsize_from_page_fraction


def counts_without_inf(number_halos):
    with np.errstate(divide="ignore", invalid="ignore"):
        number_halos_inverse = 1 / np.sqrt(number_halos)
        number_halos_inverse[np.abs(number_halos_inverse) == np.inf] = 0

    return number_halos_inverse


def monofonic_tests():
    fig: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax: Axes = fig.gca()

    linestyles = ["solid", "dotted"]
    resolutions = [128]
    if has_1024_simulations:
        resolutions.append(1024)
    else:
        resolutions.append(512)

    plank_cosmo = cosmology.cosmologies["planck18"]
    our_cosmo = {
        # "sigma8": 0.807,
        "H0": 0.67742 * 100,
        "Om0": 0.3099,
        "Ob0": 0.048891054,
        "ns": 0.96822,
    }

    cosmology.addCosmology("ourCosmo", params={**plank_cosmo, **our_cosmo})
    cosmology.setCosmology("ourCosmo")
    print(cosmology.getCurrent())
    x = None
    for i, waveform in enumerate(
            [
                "DB2", "shannon", "shannon_rehalo",
                # "resim_master", "resim_newrandom", "resim_newerrandom",
                # "resim_newswift",
                "resim_mastergadget"
            ]):
        for j, resolution in enumerate(resolutions):
            if (waveform == "shannon_rehalo" or "resim" in waveform) and resolution != 128:
                continue
            print(waveform, resolution)
            dir = base_dir / f"{waveform}_{resolution}_100"
            if "gadget" in waveform:
                halos = read_gadget_halos(dir)
                halo_masses: np.ndarray = halos["Masses"].to_numpy() * 1e10
            else:
                halos = read_velo_halos(dir)
                halo_masses: np.ndarray = halos["Mvir"].to_numpy() * 1e10 * 0.67742
            # halo_masses = halo_masses[halo_masses > 0]
            halo_masses.sort()
            print(halo_masses[-4:])

            # halos = read_halo_file(dir/"fof_output_0004.hdf5")
            # halo_masses: np.ndarray = halos["Masses"].to_numpy() * 1e10 * 0.67742

            (
                Ns,
                deltas,
                left_edges,
                number_densities,
                lower_error_limit,
                upper_error_limit,
            ) = halo_mass_function(halo_masses, sim_volume=(100 * 0.67742) ** 3)

            ax.set_xscale("log")
            ax.set_yscale("log")
            if resolution == resolutions[-1]:
                x = left_edges

            # ax.bar(centers, number_densities, width=widths, log=True, fill=False)
            name = f"{waveform} {resolution}"
            ax.step(
                left_edges,
                number_densities,
                where="post",
                color=f"C{i + 1}",
                linestyle=linestyles[j],
                label=name,
                zorder=10
            )

            # ax.fill_between(
            #     left_edges,
            #     lower_error_limit,
            #     upper_error_limit,
            #     alpha=0.5,
            #     linewidth=0,
            #     step="post",
            # )

            # break
        # break

    mfunc = mass_function.massFunction(
        x, z=0, mdef="vir", model="tinker08", q_out="dndlnM"
    )

    ax.plot(x, mfunc, label="tinker08 (vir)", color="C0")
    ax.set_xlabel("Mass ($h^{-1} M_{\odot}$)")
    ax.set_ylabel(r"$\frac{dN}{d\log M}$ ($h^{3}Mpc^{-3}$)")

    # s: GadgetHDFSnap = pynbody.load(str(base_dir / "resim_mastergadget_128_100" / "output" / "snapshot_019.hdf5"))
    # s.physical_units()
    # h3 = s.properties["h"] ** 3
    # bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(
    #     s, log_M_min=10, log_M_max=15, delta_log_M=0.1
    # )
    # ax.errorbar(
    #     bin_center, bin_counts * h3,
    #     yerr=err,
    #     fmt='o',
    #     capthick=2,
    #     elinewidth=2,
    #     color='darkgoldenrod'
    # )
    # m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(
    #     s, log_M_min=10, log_M_max=15, delta_log_M=0.1, kern="ST"
    # )
    # ax.plot(m, dn_dlogm, color='darkmagenta', linewidth=2)

    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(f"~/tmp/halo_mass_function.svg").expanduser(), transparent=True)
    plt.show()


def halo_mass_function(halo_masses, num_bins=60, sim_volume=100 ** 3):
    bins = np.geomspace(halo_masses.min(), halo_masses.max(), num_bins + 1)
    digits = np.digitize(halo_masses, bins)
    number_densities = []
    widths = []
    centers = []
    left_edges = []
    Ns = []
    deltas = []
    for bin_id in range(num_bins):
        print_progress(bin_id + 1, num_bins)
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

    return (
        Ns,
        deltas,
        left_edges,
        number_densities,
        lower_error_limit,
        upper_error_limit,
    )


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
    (
        Ns,
        deltas,
        left_edges,
        number_densities,
        lower_error_limit,
        upper_error_limit,
    ) = halo_mass_function(masses, num_bins=50, sim_volume=box_size ** 3)
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Halo Mass [$M_\\odot$]")
    ax.set_ylabel("Number Density [$\\textrm{\\#}/Mpc^3/dlogM$]")
    ax.step(left_edges, number_densities, where="post")
    plank_cosmo = cosmology.cosmologies["planck18"]
    auriga_cosmo = {
        "sigma8": 0.807,
        "H0": 70.2,
        "Om0": 0.272,
        "Ob0": 0.0455,
        "ns": 0.961,
    }
    cosmology.addCosmology("aurigaCosmo", params={**plank_cosmo, **auriga_cosmo})
    cosmology.setCosmology("aurigaCosmo")
    print(cosmology.getCurrent())
    mfunc = mass_function.massFunction(
        left_edges, z=0, mdef="vir", model="tinker08", q_out="dndlnM"
    )

    ax.plot(left_edges, mfunc)

    ax.fill_between(
        left_edges,
        lower_error_limit,
        upper_error_limit,
        alpha=0.5,
        linewidth=0,
        step="post",
    )

    plt.show()


if __name__ == "__main__":
    monofonic_tests()
    # hmf_from_rockstar_tree(Path(argv[1]))
