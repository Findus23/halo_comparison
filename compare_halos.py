import copy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy import linalg

from cic import cic_deposit
from paths import base_dir
from read_vr_files import read_velo_halo_particles
from readfiles import read_file, read_halo_file
from remap_particle_IDs import IDScaler
from threed import plotdf3d
from utils import print_progress, memory_usage


def apply_offset_to_list(value_list, offset):
    return apply_offset(np.asarray(value_list), offset)


def apply_offset(value, offset):
    box_size = 100
    half_box = box_size / 2
    value -= offset
    return ((value + half_box) % box_size) - half_box


def compare_halo_resolutions(
        ref_waveform: str, comp_waveform: str,
        reference_resolution: int, comparison_resolution: int,
        plot=False, plot3d=False, plot_cic=False,
        single=False, velo_halos=False, force=False
):
    reference_dir = base_dir / f"{ref_waveform}_{reference_resolution}_100"
    comparison_dir = base_dir / f"{comp_waveform}_{comparison_resolution}_100/"
    # the comparison_id is used as a file name for the results
    comparison_id = reference_dir.name + "_" + comparison_dir.name
    if velo_halos:
        comparison_id += "_velo"
    outfile = (base_dir / "comparisons" / comparison_id).with_suffix(".csv")
    print(f"output: {outfile}")
    if outfile.exists() and not force:
        print(outfile, "exists already")
        print("skipping")
        return

    compared_halos = []
    skip_counter = 0

    print("reading reference file")
    df_ref, ref_meta = read_file(reference_dir / "output_0004.hdf5")
    if velo_halos:
        df_ref_halo, ref_halo_lookup, ref_unbound = read_velo_halo_particles(reference_dir, recursivly=False)
    else:
        df_ref_halo = read_halo_file(reference_dir / "fof_output_0004.hdf5")

    print("reading comparison file")
    df_comp, comp_meta = read_file(comparison_dir / "output_0004.hdf5")
    if velo_halos:
        df_comp_halo, comp_halo_lookup, comp_unbound = read_velo_halo_particles(comparison_dir, recursivly=False)

    else:
        df_comp_halo = read_halo_file(comparison_dir / "fof_output_0004.hdf5")

    print("precalculating halo memberships")
    if not velo_halos:
        ref_halo_lookup = precalculate_halo_membership(df_ref, df_ref_halo)
        comp_halo_lookup = precalculate_halo_membership(df_comp, df_comp_halo)

    print(f"Memory ref: {memory_usage(df_ref):.2f} MB")
    print(f"Memory comp: {memory_usage(df_comp):.2f} MB")

    comp_halo_masses = dict(df_comp_halo["Mvir"])

    for index, original_halo in df_ref_halo.iterrows():
        print(f"{index} of {len(df_ref_halo)} original halos")
        halo_particle_ids = ref_halo_lookup[int(index)]
        ref_halo: pd.Series = df_ref_halo.loc[index]
        ref_halo_mass = ref_halo["Mvir"]
        if ref_halo["cNFW"] < 0:
            print("NEGATIVE")
            print(ref_halo["cNFW"])
            # raise ValueError()
            continue
        if len(halo_particle_ids) < 50:
            # TODO: decide on a lower size limit (and also apply it to comparison halo?)
            print(f"halo is too small ({len(halo_particle_ids)}")
            print("skipping")
            continue
        print("LEN", len(halo_particle_ids), ref_halo.Mass_tot)
        offset_x, offset_y = ref_halo.X, ref_halo.Y
        # cumulative_mass_profile(particles_in_ref_halo, ref_halo, ref_meta, plot=plot)

        prev_len = len(halo_particle_ids)
        unscaled_halo_particle_ids = copy.copy(halo_particle_ids)
        if reference_resolution < comparison_resolution:
            print("upscaling IDs")
            upscaled_ids = set()
            scaler = IDScaler(reference_resolution, comparison_resolution)
            for id in halo_particle_ids:
                upscaled_ids.update(set(scaler.upscale(id)))
            halo_particle_ids = upscaled_ids
            after_len = len(upscaled_ids)
            print(f"{prev_len} => {after_len} (factor {after_len / prev_len})")
        if comparison_resolution < reference_resolution:
            # NOTE: downscaling is a lot less efficient than upscaling
            print("downscaling IDs")
            downscaled_ids = set()
            scaler = IDScaler(comparison_resolution, reference_resolution)
            for id in halo_particle_ids:
                downscaled_ids.add(scaler.downscale(id))
            halo_particle_ids = downscaled_ids
            after_len = len(halo_particle_ids)
            print(f"{prev_len} => {after_len} (factor {prev_len / after_len:.2f})")

        halo_distances = np.linalg.norm(
            ref_halo[['X', 'Y', 'Z']].values
            - df_comp_halo[['X', 'Y', 'Z']].values,
            axis=1)
        # print(list(halo_distances))

        print(f"find nearby halos (5x{ref_halo.Rvir:.1f})")
        print(ref_halo[['X', 'Y', 'Z']].values)
        # Find IDs of halos that are less than 5 Rvir away
        nearby_halos = set(df_comp_halo.loc[halo_distances < ref_halo.Rvir * 5].index.to_list())
        if len(nearby_halos) < 10:
            print(f"only {len(nearby_halos)} halos, expanding to 50xRvir")
            nearby_halos = set(df_comp_halo.loc[halo_distances < ref_halo.Rvir * 50].index.to_list())
        if len(nearby_halos) < 10:
            print(f"only {len(nearby_halos)} halos, expanding to 150xRvir")
            nearby_halos = set(df_comp_halo.loc[halo_distances < ref_halo.Rvir * 150].index.to_list())

        if not nearby_halos:
            raise Exception("no halos are nearby")  # TODO
            # continue
        print(f"found {len(nearby_halos)} halos")
        if plot or plot3d or plot_cic or (not velo_halos):
            print("look up halo particles in comparison dataset")
            halo_particles = df_ref.loc[list(unscaled_halo_particle_ids)]

        # halos_in_particles = set(comp_halo_lookup.keys())
        # if velo_halos:
        #     ...
        #     # for halo_id, halo_set in comp_halo_lookup.items():
        #     #     if halo_particle_ids.isdisjoint(halo_set):
        #     #         continue
        #     #     halos_in_particles.add(halo_id)
        # else:
        #     halos_in_particles = set(halo_particles["FOFGroupIDs"])
        #     halos_in_particles.discard(2147483647)
        # print(f"{len(halos_in_particles)} halos found in new particles")
        # print(halos_in_particles)
        # print(halos_in_particles_alt)
        # print(halos_in_particles == halos_in_particles_alt)
        # exit()
        # assert halos_in_particles == halos_in_particles_alt
        # continue
        if plot:
            fig: Figure = plt.figure()
            ax: Axes = fig.gca()
            ax.scatter(apply_offset_to_list(halo_particles["X"], offset_x),
                       apply_offset_to_list(halo_particles["Y"], offset_y), s=1,
                       alpha=.3, label="Halo")
        if plot_cic:
            diameter = ref_halo["R_size"]
            X = ref_halo["Xc"]
            Y = ref_halo["Yc"]
            Xs = (halo_particles.X.to_numpy() - X) / diameter / 2 + 0.5
            Ys = (halo_particles.Y.to_numpy() - Y) / diameter / 2 + 0.5
            print(min(Xs), max(Xs))
            # ax.scatter(Xs, Ys)
            # plt.show()
            rho = cic_deposit(Xs, Ys, 1000)
            cmap = plt.cm.viridis
            data = np.log(1.001 + rho)
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = cmap(norm(data))
            plt.imsave(f"out_{index}.png", image)
            fig: Figure = plt.figure()
            ax: Axes = fig.gca()

            i = ax.imshow(1.001 + rho, norm=LogNorm())
            fig.colorbar(i)

            plt.show()

        if plot3d:
            from pyvista import Plotter
            pl = Plotter()
            plotdf3d(pl, halo_particles, color="#b3cde3")  # light blue
            pl.set_focus((ref_halo.X, ref_halo.Y, ref_halo.Z))
        #     ax.scatter(particles_in_ref_halo["X"], particles_in_ref_halo["Y"], s=1, alpha=.3, label="RefHalo")
        # plt.legend()
        # plt.show()
        best_halo = None
        best_halo_match = 0
        num_skipped_for_mass = 0
        for i, halo_id in enumerate(nearby_halos):
            # print("----------", halo, "----------")
            # halo_data = df_comp_halo.loc[halo]
            # particles_in_comp_halo: DataFrame = df_comp.loc[df_comp["FOFGroupIDs"] == halo]
            particle_ids_in_comp_halo = comp_halo_lookup[halo_id]
            mass_factor_limit = 5

            if not (1 / mass_factor_limit < (comp_halo_masses[halo_id] / ref_halo_mass) < mass_factor_limit):
                # print("mass not similar, skipping")
                num_skipped_for_mass += 1
                continue

            halo_size = len(particle_ids_in_comp_halo)
            # df = particles_in_comp_halo.join(halo_particles, how="inner", rsuffix="ref")
            shared_particles = particle_ids_in_comp_halo.intersection(halo_particle_ids)
            union_particles=particle_ids_in_comp_halo.union(halo_particle_ids)

            shared_size = len(shared_particles)/len(union_particles)
            # print(shared_size)
            if not shared_size:
                continue

            if plot or plot3d:
                df = df_comp.loc[list(shared_particles)]
            if plot:
                color = f"C{i + 1}"
                comp_halo: pd.Series = df_comp_halo.loc[halo_id]

                ax.scatter(apply_offset_to_list(df["X"], offset_x), apply_offset_to_list(df["Y"], offset_y), s=1,
                           alpha=.3, c=color)
                circle = Circle((apply_offset(comp_halo.X, offset_x), apply_offset(comp_halo.Y, offset_y)),
                                comp_halo["Rvir"], zorder=10,
                                linewidth=1, edgecolor=color, fill=None
                                )
                ax.add_artist(circle)
            if plot3d:
                plotdf3d(pl, df, color="#fed9a6")  # light orange
            if shared_size > best_halo_match:
                best_halo_match = shared_size
                best_halo = halo_id
        print(f"skipped {num_skipped_for_mass} halos due to mass ratio")
        if not best_halo:
            skip_counter += 1
            continue
        comp_halo: pd.Series = df_comp_halo.loc[best_halo]

        # merge the data of the two halos with fitting prefixes
        halo_data = pd.concat([
            ref_halo.add_prefix("ref_"),
            comp_halo.add_prefix("comp_")
        ])
        distance = linalg.norm(
            np.array([ref_halo.X, ref_halo.Y, ref_halo.Z]) - np.array([comp_halo.X, comp_halo.Y, comp_halo.Z])
        ) / ref_halo.Rvir
        halo_data["distance"] = distance
        halo_data["match"] = best_halo_match
        compared_halos.append(halo_data)
        # exit()
        if plot:
            print(f"plotting with offsets ({offset_x},{offset_y})")
            # ax.legend()
            ax.set_title(f"{reference_dir.name} vs. {comparison_dir.name} (Halo {index})")
            fig.savefig("out.png", dpi=300)
            plt.show()
        if plot3d:
            pl.show()
        if single:
            break

    df = pd.concat(compared_halos, axis=1).T
    print(df)
    print(f"saving to {outfile}")
    df.to_csv(outfile, index=False)
    df.to_hdf(outfile.with_suffix(".hdf5"), key="comparison", complevel=5)
    print(skip_counter)
    return df, reference_dir.name + "_" + comparison_dir.name


def precalculate_halo_membership(df_comp, df_comp_halo):
    pointer = 0
    comp_halo_lookup: Dict[int, set[int]] = {}
    for i, halo in df_comp_halo.iterrows():
        print_progress(i, len(df_comp_halo), halo["Sizes"])
        size = int(halo["Sizes"])
        halo_id = int(i)
        halo_particles = df_comp.iloc[pointer:pointer + size]

        # check_id = halo_particles["FOFGroupIDs"].to_numpy()
        # assert (check_id == i).all()
        # assert (check_id==check_id[0]
        pointer += size
        ids = set(halo_particles.index.to_list())
        comp_halo_lookup[halo_id] = ids
    return comp_halo_lookup


if __name__ == '__main__':
    compare_halo_resolutions(
        ref_waveform="shannon",
        comp_waveform="shannon",
        reference_resolution=128,
        comparison_resolution=256,
        plot=False,
        plot3d=False,
        plot_cic=False,
        velo_halos=True,
        single=False,
        force=True
    )
