import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from cic import cic_from_radius, cic_range
from halo_mass_profile import halo_mass_profile
from nfw import fit_nfw
from paths import auriga_dir, richings_dir
from ramses import load_ramses_data
from readfiles import read_file, read_halo_file, ParticlesMeta
from slices import create_2d_slice
from utils import read_swift_config, print_wall_time, figsize_from_page_fraction


class Mode(Enum):
    richings = 1
    auriga6 = 2


mode = Mode.richings


def dir_name_to_parameter(dir_name: str):
    return map(
        int,
        dir_name.lstrip("auriga6_halo")
        .lstrip("richings21_")
        .lstrip("bary_")
        .lstrip("ramses_")
        .split("_"),
    )


def levelmax_to_softening_length(levelmax: int) -> float:
    box_size = 100
    return box_size / 30 / 2 ** levelmax


def main():
    fig1: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax1: Axes = fig1.gca()
    fig2: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax2: Axes = fig2.gca()

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"R [Mpc]")
    ax1.set_ylabel(r"M [$10^{10} \mathrm{M}_\odot$]")
    ax2.set_ylabel("density [$\\frac{10^{10} \\mathrm{M}_\\odot}{Mpc^3}$]")

    part_numbers = []

    reference_file = Path(f"auriga_reference_{mode}.pickle")

    centers = {}

    @dataclass
    class Result:
        title: str
        rho: np.ndarray
        levels: Tuple[int, int, int]

    images = []
    vmin = np.Inf
    vmax = -np.Inf
    root_dir = auriga_dir if mode == Mode.auriga6 else richings_dir
    i = 0
    for dir in sorted(root_dir.glob("*")):
        if not dir.is_dir() or "bak" in dir.name:
            continue
        is_ramses = "ramses" in dir.name
        has_baryons = "bary" in dir.name or is_ramses
        is_by_adrian = "arj" in dir.name

        print(dir.name)

        if not is_by_adrian:
            levelmin, levelmin_TF, levelmax = dir_name_to_parameter(dir.name)
            print(levelmin, levelmin_TF, levelmax)
            if not has_baryons:
                continue
            if levelmax != 11:
                continue

        input_file = dir / "output_0009.hdf5"
        if mode == Mode.richings:
            input_file = dir / "output_0004.hdf5"
        if is_by_adrian or is_ramses:
            input_file = dir / "output_0000.hdf5"
            softening_length = None
        else:
            try:
                swift_conf = read_swift_config(dir)
                print_wall_time(dir)
            except FileNotFoundError:
                continue
            gravity_conf = swift_conf["Gravity"]
            softening_length = gravity_conf["comoving_DM_softening"]
            assert softening_length == gravity_conf["max_physical_DM_softening"]
            if "max_physical_baryon_softening" in gravity_conf:
                assert softening_length == gravity_conf["max_physical_baryon_softening"]
                assert softening_length == gravity_conf["comoving_baryon_softening"]

            ideal_softening_length = levelmax_to_softening_length(levelmax)
            if not np.isclose(softening_length, levelmax_to_softening_length(levelmax)):
                raise ValueError(
                    f"softening length for levelmax {levelmax} should be {ideal_softening_length} "
                    f"but is {softening_length}"
                )
        print(input_file)
        if mode == Mode.richings and is_by_adrian:
            h = 0.6777
            with h5py.File(dir / "Richings_object_z0.h5") as f:
                df = pd.DataFrame(f["Coordinates"][:] / h, columns=["X", "Y", "Z"])
            particles_meta = ParticlesMeta(particle_mass=1.1503e7 / 1e10)
            center = np.array([60.7, 29, 64]) / h
            softening_length = None
        elif "ramses" in dir.name:
            h = 0.6777
            hr_coordinates, particles_meta, center = load_ramses_data(dir / "output_00007")
            df = pd.DataFrame(hr_coordinates, columns=["X", "Y", "Z"])
            center = center
            softening_length = None
        else:
            df, particles_meta = read_file(input_file)
            df_halos = read_halo_file(input_file.with_name("fof_" + input_file.name))
            # vr_halo = read_velo_halos(dir, veloname="velo_out").loc[1]
            # particles_in_halo = df.loc[df["FOFGroupIDs"] == 3]

            halo_id = 1
            while True:
                particles_in_halo = df.loc[df["FOFGroupIDs"] == halo_id]
                if len(particles_in_halo) > 1:
                    break
                halo_id += 1

            halo = df_halos.loc[halo_id]
            part_numbers.append(len(df) * particles_meta.particle_mass)
            # halo = halos.loc[1]
            center = np.array([halo.X, halo.Y, halo.Z])
        log_radial_bins, bin_masses, bin_densities, center = halo_mass_profile(
            df, center, particles_meta, plot=False, num_bins=100, vmin=0.002, vmax=6.5
        )
        i_min_border = np.argmax(
            0.01 < log_radial_bins
        )  # first bin outside of specific radius
        i_max_border = np.argmax(1.5 < log_radial_bins)
        popt = fit_nfw(
            log_radial_bins[i_min_border:i_max_border],
            bin_densities[i_min_border:i_max_border],
        )  # = rho_0, r_s
        print(popt)
        # # Plot NFW profile
        # ax.loglog(
        #     log_radial_bins[i_min_border:i_max_border],
        #     nfw(log_radial_bins[i_min_border:i_max_border], *popt),
        #     linestyle="dotted"
        # )

        centers[dir.name] = center
        if is_by_adrian:
            with reference_file.open("wb") as f:
                pickle.dump([log_radial_bins, bin_masses, bin_densities], f)
        if is_by_adrian:
            label = "reference"
        else:
            label = f"{levelmin}, {levelmin_TF}, {levelmax}"
        ax1.loglog(log_radial_bins[:-1], bin_masses, label=label, c=f"C{i}")

        ax2.loglog(log_radial_bins[:-1], bin_densities, label=label, c=f"C{i}")

        if reference_file.exists() and not is_by_adrian:
            with reference_file.open("rb") as f:
                data: List[np.ndarray] = pickle.load(f)
                ref_log_radial_bins, ref_bin_masses, ref_bin_densities = data
            mass_deviation: np.ndarray = np.abs(bin_masses - ref_bin_masses)
            density_deviation: np.ndarray = np.abs(bin_densities - ref_bin_densities)
            ax1.loglog(log_radial_bins[:-1], mass_deviation, c=f"C{i}", linestyle="dotted")

            ax2.loglog(
                log_radial_bins[:-1], density_deviation, c=f"C{i}", linestyle="dotted"
            )
            accuracy = mass_deviation / ref_bin_masses
            print(accuracy)
            print("mean accuracy", accuracy.mean())

        if softening_length:
            for ax in [ax1, ax2]:
                ax.axvline(4 * softening_length, color=f"C{i}", linestyle="dotted")
        # for ax in [ax1, ax2]:
        #     ax.axvline(vr_halo.Rvir, color=f"C{i}", linestyle="dashed")

        X, Y, Z = df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy()

        # shift: (-6, 0, -12)
        # if not is_by_adrian:
        #     xshift = Xc - Xc_adrian
        #     yshift = Yc - Yc_adrian
        #     zshift = Zc - Zc_adrian
        #     print("shift", xshift, yshift, zshift)

        X -= center[0]
        Y -= center[1]
        Z -= center[2]

        rho, extent = cic_from_radius(X, Z, 4000, 0, 0, 5, periodic=False)

        vmin = min(vmin, rho.min())
        vmax = max(vmax, rho.max())

        images.append(
            Result(
                rho=rho,
                title=str(dir.name),
                levels=(levelmin, levelmin_TF, levelmax) if levelmin else None,
            )
        )
        i += 1

        if has_baryons:
            fig3, axs_baryon = plt.subplots(nrows=1, ncols=5, sharex="all", sharey="all", figsize=(10, 4))
            extent = [46, 52, 54, 60]  # xrange[0], xrange[-1], yrange[0], yrange[-1]
            for ii, property in enumerate(["cic", "Densities", "Entropies", "InternalEnergies", "Temperatures"]):
                print(property)
                if property == "cic":
                    grid, _ = cic_range(X + center[0], Y + center[1], 1000, *extent, periodic=False)
                    grid = grid.T
                else:
                    grid = create_2d_slice(input_file, center, property=property, extent=extent)
                print("minmax", grid.min(), grid.max())
                assert grid.min() != grid.max()
                ax_baryon: Axes = axs_baryon[ii]
                img = ax_baryon.imshow(
                    grid,
                    norm=LogNorm(),
                    interpolation="none",
                    origin="lower",
                    extent=extent,
                )
                ax_baryon.set_title(property)
                # ax_baryon.set_xlabel("X")
                # ax_baryon.set_ylabel("Y")
                ax_baryon.set_aspect("equal")
            fig3.suptitle(input_file.parent.stem)
            fig3.tight_layout()
            fig3.savefig(Path("~/tmp/slice.png").expanduser(), dpi=300)
            plt.show()

        # plot_cic(
        #     rho, extent,
        #     title=str(dir.name)
        # )
    ax1.legend()
    ax2.legend()
    fig1.tight_layout()
    fig2.tight_layout()

    # fig3: Figure = plt.figure(figsize=(9, 9))
    # axes: List[Axes] = fig3.subplots(3, 3, sharex=True, sharey=True).flatten()
    fig3: Figure = plt.figure(
        figsize=figsize_from_page_fraction(columns=2, height_to_width=1)
    )
    axes: List[Axes] = fig3.subplots(3, 3, sharex=True, sharey=True).flatten()

    for result, ax in zip(images, axes):
        data = 1.1 + result.rho
        vmin_scaled = 1.1 + vmin
        vmax_scaled = 1.1 + vmax
        img = ax.imshow(
            data.T,
            norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled),
            extent=extent,
            origin="lower",
        )
        ax.set_title(result.title)

    fig3.tight_layout()
    fig3.subplots_adjust(right=0.825)
    cbar_ax = fig3.add_axes([0.85, 0.05, 0.05, 0.9])
    fig3.colorbar(img, cax=cbar_ax)

    fig1.savefig(Path(f"~/tmp/auriga1.pdf").expanduser())
    fig2.savefig(Path(f"~/tmp/auriga2.pdf").expanduser())
    fig3.savefig(Path("~/tmp/auriga3.pdf").expanduser())
    pprint(centers)
    plt.show()
    print(part_numbers)


if __name__ == '__main__':
    main()
