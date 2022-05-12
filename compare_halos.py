from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import linalg
from pandas import DataFrame
from pyvista import Plotter

from paths import base_dir
from read_vr_files import read_velo_halos
from readfiles import read_file, read_halo_file
from remap_particle_IDs import IDScaler
from threed import plotdf3d
from utils import print_progress, memory_usage


def apply_offset_to_list(value_list, offset):
    result_list = []
    for value in value_list:
        value = apply_offset(value, offset)
        result_list.append(value)
    return result_list


def apply_offset(value, offset):
    box_size = 100
    if value > box_size / 2:
        value -= box_size
    value -= offset
    return value


def compare_halo_resolutions(reference_resolution: int, comparison_resolution: int,
                             plot=False, plot3d=False, single=False, velo_halos=False):
    reference_dir = base_dir / f"shannon_{reference_resolution}_100"
    comparison_dir = base_dir / f"shannon_{comparison_resolution}_100/"
    comparison_id = reference_dir.name + "_" + comparison_dir.name
    if velo_halos:
        comparison_id += "_velo"
    ref_masses = []
    comp_masses = []
    ref_sizes = []
    comp_sizes = []
    matches = []
    distances = []
    skip_counter = 0

    print("reading reference file")
    df_ref, ref_meta = read_file(reference_dir)
    if velo_halos:
        df_ref_halo, ref_halo_lookup, ref_unbound = read_velo_halos(reference_dir, recursivly=False)
        for k, v in ref_halo_lookup.items():
            v.update(ref_unbound[k])
    else:
        df_ref_halo = read_halo_file(reference_dir)

    print("reading comparison file")
    df_comp, comp_meta = read_file(comparison_dir)
    if velo_halos:
        df_comp_halo, comp_halo_lookup, comp_unbound = read_velo_halos(comparison_dir, recursivly=False)
        for k, v in comp_halo_lookup.items():
            v.update(comp_unbound[k])

    else:
        df_comp_halo = read_halo_file(comparison_dir)

    print("precalculating halo memberships")
    if not velo_halos:
        ref_halo_lookup = precalculate_halo_membership(df_ref, df_ref_halo)
        comp_halo_lookup = precalculate_halo_membership(df_comp, df_comp_halo)

    print(f"Memory ref: {memory_usage(df_ref):.2f} MB")
    print(f"Memory comp: {memory_usage(df_comp):.2f} MB")

    for index, original_halo in df_ref_halo.iterrows():
        print(f"{index} of {len(df_ref_halo)} original halos")
        halo_particle_ids = ref_halo_lookup[int(index)]
        ref_halo = df_ref_halo.loc[index]
        print("LEN", len(halo_particle_ids), ref_halo.Mass_tot)
        if 1 < len(halo_particle_ids) < 20:
            raise ValueError("test")
            print("skipping")
            continue
        if not halo_particle_ids:
            continue
        offset_x, offset_y = ref_halo.X, ref_halo.Y
        # cumulative_mass_profile(particles_in_ref_halo, ref_halo, ref_meta, plot=plot)

        prev_len = len(halo_particle_ids)
        if reference_resolution < comparison_resolution:
            print("upscaling IDs")
            upscaled_ids = set()
            scaler = IDScaler(reference_resolution, comparison_resolution)
            for id in halo_particle_ids:
                upscaled_ids.update(set(scaler.upscale(id)))
            halo_particle_ids = upscaled_ids
            after_len = len(upscaled_ids)
            print(prev_len, after_len)
            print(f"{prev_len} => {after_len} (factor {after_len / prev_len})")
        if comparison_resolution < reference_resolution:
            print("downscaling IDs")
            downscaled_ids = set()
            scaler = IDScaler(comparison_resolution, reference_resolution)
            for id in halo_particle_ids:
                downscaled_ids.add(scaler.downscale(id))
            halo_particle_ids = downscaled_ids
            after_len = len(halo_particle_ids)
            print(f"{prev_len} => {after_len} (factor {prev_len / after_len})")

        print("look up halo particles in comparison dataset")

        halo_distances = np.linalg.norm(
            ref_halo[['X', 'Y', 'Z']].values
            - df_comp_halo[['X', 'Y', 'Z']].values,
            axis=1)
        # print(list(halo_distances))

        nearby_halos = set(df_comp_halo.loc[halo_distances < ref_halo.Rvir * 5].index.to_list())

        if plot or plot3d or (not velo_halos):
            halo_particles = df_comp.loc[list(halo_particle_ids)]

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
            halo_particles.to_csv(f"halo{index}.csv")
            ax.scatter(apply_offset_to_list(halo_particles["X"], offset_x),
                       apply_offset_to_list(halo_particles["Y"], offset_y), s=1,
                       alpha=.3, label="Halo")
        if plot3d:
            pl = Plotter()
            plotdf3d(pl, halo_particles, color="#b3cde3")  # light blue
            pl.set_focus((ref_halo.X, ref_halo.Y, ref_halo.Z))
        #     ax.scatter(particles_in_ref_halo["X"], particles_in_ref_halo["Y"], s=1, alpha=.3, label="RefHalo")
        # plt.legend()
        # plt.show()
        best_halo = None
        best_halo_match = 0
        if not nearby_halos:
            continue
            raise Exception("something doesn't make any sense")  # TODO
            # continue

        for i, halo_id in enumerate(nearby_halos):
            # print("----------", halo, "----------")
            # halo_data = df_comp_halo.loc[halo]
            # particles_in_comp_halo: DataFrame = df_comp.loc[df_comp["FOFGroupIDs"] == halo]
            particle_ids_in_comp_halo = comp_halo_lookup[halo_id]
            halo_size = len(particle_ids_in_comp_halo)
            # df = particles_in_comp_halo.join(halo_particles, how="inner", rsuffix="ref")
            shared_particles = particle_ids_in_comp_halo.intersection(halo_particle_ids)
            shared_size = len(shared_particles)
            # print(shared_size)
            if not shared_size:
                continue
            size_match = shared_size / halo_size

            # if shared_size==halo_size:
            #     raise Exception("match")
            if plot or plot3d:
                df = df_comp.loc[list(shared_particles)]
            if plot:
                color = f"C{i + 1}"

                ax.scatter(apply_offset_to_list(df["X"], offset_x), apply_offset_to_list(df["Y"], offset_y), s=1,
                           alpha=.3, c=color)
                comp_halo = df_comp_halo.loc[halo_id]
                # circle = Circle((apply_offset(comp_halo.X, offset_x), apply_offset(comp_halo.Y, offset_y)),
                #                 comp_halo["Sizes"] / 1000, zorder=10,
                #                 linewidth=1, edgecolor=color, fill=None
                #                 )
                # ax.add_artist(circle)
            if plot3d:
                plotdf3d(pl, df, color="#fed9a6")  # light orange
            # print_progress(i, len(halos_in_particles), halo)
            # ax.scatter(particles_in_comp_halo["X"], particles_in_comp_halo["Y"], s=2, alpha=.3, label=f"shared {halo}")
            if shared_size > best_halo_match:
                best_halo_match = shared_size
                best_halo = halo_id

        if not best_halo:
            skip_counter += 1
            continue
        comp_halo = df_comp_halo.loc[best_halo]

        print(ref_halo)
        print(comp_halo)
        if velo_halos:
            ref_sizes.append(ref_halo.Rvir)
            comp_sizes.append(comp_halo.Rvir)
            ref_masses.append(ref_halo.Mass_tot)
            comp_masses.append(comp_halo.Mass_tot)
        else:
            ref_sizes.append(0)
            ref_masses.append(ref_halo["Masses"])
            comp_sizes.append(0)
            comp_masses.append(comp_halo["Masses"])
        distances.append(linalg.norm(
            np.array([ref_halo.X, ref_halo.Y, ref_halo.Z]) - np.array([comp_halo.X, comp_halo.Y, comp_halo.Z])
        ) / ref_halo.Rvir)
        matches.append(best_halo_match / len(halo_particle_ids))
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

    df = DataFrame(np.array([matches, distances, ref_sizes, comp_sizes, ref_masses, comp_masses]).T,
                   columns=["matches", "distances", "ref_sizes", "comp_sizes", "ref_masses", "comp_masses"])
    print(df)
    outfile = comparison_id + ".csv"
    print(f"saving to {outfile}")
    df.to_csv(comparison_id + ".csv", index=False)
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
        reference_resolution=128,
        comparison_resolution=512,
        plot=False,
        plot3d=False,
        velo_halos=True,
        single=False
    )
