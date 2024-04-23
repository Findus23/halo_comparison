import json
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint
from subprocess import run
from sys import argv
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import pynbody
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy import log10
from pynbody.array import SimArray
from pynbody.snapshot import FamilySubSnap
from pynbody.snapshot.ramses import RamsesSnap
from scipy import constants

from cache import HDFCache
from cic import cic_from_radius, cic_range
from find_center import find_center
from halo_mass_profile import halo_mass_profile, property_profile
from nfw import fit_nfw
from paths import auriga_dir, richings_dir, auriga_dir_new, richings_dir_new
from ramses import load_ramses_data, get_slice_argument, load_slice_data
from read_vr_files import read_velo_halos
from readfiles import read_file, read_halo_file, ParticlesMeta
from slices import create_2d_slice, filter_3d
from utils import read_swift_config, figsize_from_page_fraction


class Mode(Enum):
    richings = "richings"
    auriga6 = "auriga"


mode = Mode(argv[1])

cache = HDFCache(Path("auriga_cache.hdf5"))


def dir_name_to_parameter(dir_name: str):
    return map(
        int,
        dir_name.lstrip("auriga6_halo")
        .lstrip("auriga6_music20_")
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
    axs_baryon: List[List[Axes]]
    fig4: Figure
    fig4, axs_baryon = plt.subplots(
        nrows=2, ncols=4,
        sharex="all", sharey="all",
        figsize=figsize_from_page_fraction(columns=2, height_to_width=0.55),
        layout='constrained'
    )
    fig5: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax5: Axes = fig5.gca()
    fig6: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax6: Axes = fig6.gca()
    baryon_plot_counter = 0
    vminmax = {}
    extents = []

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"R [Mpc]")
    ax1.set_ylabel(r"M [$10^{10} \mathrm{M}_\odot$]")
    ax2.set_ylabel("density [$\\frac{10^{10} \\mathrm{M}_\\odot}{Mpc^3}$]")

    part_numbers = []

    reference_file = Path(f"auriga_reference_{mode.value}.pickle")

    centers = {}
    halo_props = {
        "R_200crit": {},
        "Mass_200crit": {},
        "cNFW_200crit": {},
        "FOF_Mass": {},
        "FOF_Size": {},
    }
    halo_props["Mass_200crit"]["Au6_paper"] = 104.39
    halo_props["R_200crit"]["Au6_paper"] = 213.83 / 1000

    @dataclass
    class Result:
        title: str
        rho: np.ndarray
        levels: Tuple[int, int, int]

    images = []
    vmin = np.Inf
    vmax = -np.Inf
    root_dir = auriga_dir if mode == Mode.auriga6 else richings_dir
    if True:
        root_dir = auriga_dir_new if mode == Mode.auriga6 else richings_dir_new
    i = 0
    mapping = {}
    for dir in sorted(root_dir.glob("*")):
        dir = dir.resolve()
        print("----------------------------")
        if not dir.is_dir() or "bak" in dir.name:
            continue
        is_ramses = "ramses" in dir.name
        has_baryons = "bary" in dir.name or is_ramses
        is_by_adrian = "arj" in dir.name
        is_resim = dir.name[0].isdigit() or is_by_adrian

        print(dir.name)

        if not is_by_adrian:
            levelmin, levelmin_TF, levelmax = dir_name_to_parameter(dir.name)
            print(levelmin, levelmin_TF, levelmax)
            #   if not is_resim:
            #      continue
            # if levelmax < 12 and not is_by_adrian:
            #     continue
            if mode == Mode.auriga6:
                print("sdfds")
                if (levelmin, levelmin_TF, levelmax) == (7, 9, 9):
                    print("Aaaa")
                    continue
            elif mode == Mode.richings:
                if not has_baryons:
                    continue
                # if levelmax != 11:
                #     continue
            # if not is_ramses:
            #     continue
            if dir.name in ["bary_ramses_7_10_11"]:
                continue
        input_file = dir / "output_0007.hdf5"
        print(input_file)
        if mode == Mode.richings and not is_resim:
            input_file = dir / "output_0004.hdf5"
        if is_by_adrian or is_ramses:
            input_file = dir / "output_0000.hdf5"
            softening_length = None
        else:
            try:
                swift_conf = read_swift_config(dir)
                # print_wall_time(dir)
            except FileNotFoundError:
                print(f"file not found: {dir}")
                continue
            gravity_conf = swift_conf["Gravity"]
            softening_length = gravity_conf["comoving_DM_softening"]
            assert softening_length == gravity_conf["max_physical_DM_softening"]
            if "max_physical_baryon_softening" in gravity_conf:
                assert softening_length == gravity_conf["max_physical_baryon_softening"]
                assert softening_length == gravity_conf["comoving_baryon_softening"]

            ideal_softening_length = levelmax_to_softening_length(levelmax)
            if not np.isclose(softening_length, ideal_softening_length):
                if softening_length < ideal_softening_length:
                    print(
                        f"softening length smaller than calculated: {softening_length}<{ideal_softening_length} ({levelmax=})")
                else:
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
            hr_coordinates, particles_meta, center = load_ramses_data(dir / "output_00009")
            df = pd.DataFrame(hr_coordinates, columns=["X", "Y", "Z"])
            softening_length = None
        else:
            df, particles_meta = read_file(input_file)
            df_halos = read_halo_file(input_file.with_name("fof_" + input_file.name))

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
            halo_props["FOF_Mass"][dir.name] = halo["Masses"]
            halo_props["FOF_Size"][dir.name] = halo["Sizes"]

            if mode == Mode.auriga6:
                vr_halo = read_velo_halos(dir, veloname="velo_out").loc[1]
                print(vr_halo["R_200crit"])
                halo_props["R_200crit"][dir.name] = vr_halo["R_200crit"]
                halo_props["Mass_200crit"][dir.name] = vr_halo["Mass_200crit"]
                halo_props["cNFW_200crit"][dir.name] = vr_halo["cNFW_200crit"]
                center2 = np.array([vr_halo.X, vr_halo.Y, vr_halo.Z])
                print("center-diff", center2 - center)
                print("center-diff", center)
                print("center-diff", center2)

        center = find_center(df, center)
        log_radial_bins, bin_masses, bin_densities, center = halo_mass_profile(
            df[["X", "Y", "Z"]].to_numpy(), center, particles_meta, plot=False,
            num_bins=100, rmin=0.002, rmax=6.5
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
            label = "Reference"
        else:
            label = f"({levelmin}, {levelmin_TF}, {levelmax})"
        ax1.loglog(log_radial_bins[:-1], bin_masses, label=label, c=f"C{i}")

        ax2.loglog(log_radial_bins[:-1], bin_densities, label=label, c=f"C{i}")

        mapping[i] = dir.name

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
        coords = df[["X", "Y", "Z"]].to_numpy()

        # shift: (-6, 0, -12)
        # if not is_by_adrian:
        #     xshift = Xc - Xc_adrian
        #     yshift = Yc - Yc_adrian
        #     zshift = Zc - Zc_adrian
        #     print("shift", xshift, yshift, zshift)

        coords_centered = coords - center

        rho, extent = cic_from_radius(coords_centered[::, 0], coords_centered[::, 2], 500, 0, 0, 1.5, periodic=False)

        vmin = min(vmin, rho.min())
        vmax = max(vmax, rho.max())
        res = Result(
            rho=rho,
            title=f"levelmin={levelmin}, levelmin_TF={levelmin_TF}, levelmax={levelmax}" if not is_by_adrian else "Reference",
            levels=(levelmin, levelmin_TF, levelmax) if not is_by_adrian else (100, 100, 100),
        )
        res.title = res.title + "\n" + dir.name
        images.append(res)
        i += 1

        if has_baryons:
            interpolation_method = "nearest"  # "linear"
            bary_file = dir / "output_00009" if is_ramses else input_file
            if is_ramses:
                s: RamsesSnap = pynbody.load(str(bary_file))
                gas_data: FamilySubSnap = s.gas
                temperature_array: SimArray = gas_data["temp"]
                p_array: SimArray = gas_data["p"].in_units("1e10 Msol Mpc^-3 km^2 s^-2")
                rho_array: SimArray = gas_data["rho"].in_units("1e10 Msol Mpc^-3")
                coord_array: SimArray = gas_data["pos"].in_units("Mpc")
                mass_array = np.asarray(gas_data["mass"].in_units("1e10 Msol"))
                bary_coords = np.asarray(coord_array)
                bary_properties = {
                    "Temperatures": np.asarray(temperature_array.in_units("K")),
                    "Pressures": np.asarray(p_array),
                    "Densities": np.asarray(rho_array),
                    "Entropies": np.asarray(log10(p_array / rho_array ** (5 / 3))),
                }
            else:
                with h5py.File(input_file) as f:
                    pt0 = f["PartType0"]
                    bary_coords = pt0["Coordinates"][:]
                    mass_array = pt0["Masses"][:]
                    bary_properties = {
                        "InternalEnergies": pt0["InternalEnergies"][:],
                        "Densities": pt0["Densities"][:],
                        "Pressures": pt0["Pressures"][:],
                        # "Entropies": log10(pt0["Densities"][:] / pt0["Densities"][:] ** (5 / 3)),
                        "Entropies": pt0["Entropies"][:]
                    }
                    bary_properties["Temperatures"] = bary_properties["InternalEnergies"]

            radius = 1.9
            resolution = 500
            # xrange[0], xrange[-1], yrange[0], yrange[-1]
            extent = [center[0] - radius, center[0] + radius,
                      center[1] - radius, center[1] + radius]
            extents.append(extent)
            # extent = [42, 62, 50, 70]
            ramses_done = False
            labels = {
                "cic": "$\\rho_{DM}$",
                "Densities": "$\\rho$",
                "Entropies": "$s$",
                "Temperatures": "$T$",
            }
            for ii, property in enumerate(["cic", "Densities", "Entropies", "Temperatures"]):
                print("property:", property)
                key = f"grid_{resolution}_{property}_{interpolation_method}_{radius}"
                cached_grid = cache.get(key, str(bary_file))
                if cached_grid is not None:
                    grid = cached_grid
                else:
                    print("grid not yet cached, calculating now")
                    if property == "cic":
                        coords_in_box = filter_3d(coords, extent, zlimit=(center[2] - .1, center[2] + .1))
                        rho, _ = cic_range(coords_in_box[::, 0], coords_in_box[::, 1], resolution, *extent,
                                           periodic=False)
                        grid = 1.1 + rho.T
                    else:
                        if not is_ramses:
                            grid = create_2d_slice(center, coords=bary_coords,
                                                   resolution=resolution,
                                                   property_name=property,
                                                   property_data=bary_properties[property],
                                                   extent=extent, method=interpolation_method)
                        else:
                            frac_center = center / 100
                            frac_extent = np.array(extent) / 100

                            print(frac_extent)
                            print(frac_center)
                            args, imager_dir = get_slice_argument(
                                frac_extent, frac_center,
                                bary_file, interpolation_method,
                                depth=.001
                            )
                            print(" ".join(args))
                            if not ramses_done:
                                run(args, cwd=imager_dir)
                                ramses_done = True
                            property_map = {
                                "Densities": "rhomap",
                                "Entropies": "Smap",
                                "Temperatures": "Tmap"
                            }

                            fname = imager_dir / f"snapshot_{property_map[property]}_zproj_zobs-0p00.bin"
                            grid = load_slice_data(fname).T

                    cache.set(key, grid, str(bary_file), compressed=True)
                if property == "Densities" and is_ramses:
                    # convert g/cm^3 to 1e10 Msol Mpc^-3
                    solar_mass_in_gram = 1.988e33
                    mpc_in_cm = constants.parsec * constants.mega * 100
                    grid = np.asarray(grid)
                    grid *= (mpc_in_cm ** 3 / solar_mass_in_gram / 1e10)

                ax_baryon = axs_baryon[baryon_plot_counter][ii]
                if property in vminmax:
                    minmax = vminmax[property]
                else:
                    minmax = np.min(grid), np.max(grid)
                    vminmax[property] = minmax
                print("minmax", minmax)
                img: AxesImage = ax_baryon.imshow(
                    grid,
                    norm=LogNorm(vmin=minmax[0], vmax=minmax[1]),
                    interpolation="none",
                    origin="lower",
                    extent=extent,
                )

                if baryon_plot_counter == 0:
                    fig4.colorbar(img, ax=ax_baryon, location="top", label=labels[property])
                    # ax_baryon.set_title(property)
                # ax_baryon.set_xlabel("X")
                # ax_baryon.set_ylabel("Y")
                ax_baryon.set_aspect("equal")
            # exit()
            baryon_plot_counter += 1
            continue

            r, prof = property_profile(bary_coords, center, mass_array, bary_properties, num_bins=100, rmin=0.002,
                                       rmax=6.5)
            integrator_name = "Ramses" if is_ramses else "Swift"
            label = f"{integrator_name} {levelmin}, {levelmin_TF}, {levelmax}"
            ax5.set_title("Densities")
            ax6.set_title("Pressures")
            ax5.loglog(r[1:], prof["Densities"], label=label)
            ax6.loglog(r[1:], prof["Pressures"], label=label)

    if mode==Mode.richings:
        extents = np.asarray(extents)
        print(extents)
        global_xlim = extents[:, 0].max(), extents[:, 1].min()
        global_ylim = extents[:, 2].max(), extents[:, 3].min()

        for ax in axs_baryon.flatten():
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)

    fig3: Figure = plt.figure(
        # just a bit more than 2/3 so that the two rows don't overlap
        figsize=figsize_from_page_fraction(columns=2, height_to_width=33 / 48)
    )
    axes: List[Axes] = fig3.subplots(2, 3, sharex="all", sharey="all").flatten()
    # images.sort(key=lambda r: r.levels, reverse=True)

    for result, ax in zip(images, axes):
        data = 1.1 + result.rho
        vmin_scaled = 1.1 + vmin
        vmax_scaled = 1.1 + vmax
        img = ax.imshow(
            data.T,
            norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled),
            extent=extent,
            origin="lower",
            cmap="Greys",
            interpolation="none"
        )
        ax.text(
            0.5,
            0.95,
            result.title,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    for ax in [ax1, ax2, ax5, ax6]:
        ax.legend()
    for fig in [fig1, fig2, fig3, fig5, fig6]:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
    axs_baryon[0][0].set_ylabel("Swift")
    axs_baryon[1][0].set_ylabel("Ramses")

    fig1.savefig(Path(f"~/tmp/{mode.value}1.pdf").expanduser())
    fig2.savefig(Path(f"~/tmp/{mode.value}2.pdf").expanduser())
    fig3.savefig(Path(f"~/tmp/{mode.value}3.pdf").expanduser())

    fig4.savefig(Path(f"~/tmp/{mode.value}4.pdf").expanduser())

    pprint(centers)
    if mode == Mode.auriga6:
        with open("halo_props.json","w") as f:
            json.dump(halo_props,f, indent=4)
    plt.show()
    print(part_numbers)
    print(mapping)


if __name__ == '__main__':
    main()
