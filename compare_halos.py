from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from cumulative_mass_profiles import cumulative_mass_profile
from readfiles import read_file, read_halo_file
from remap_particle_IDs import IDScaler
from utils import print_progress, memory_usage


def compare_halo_resolutions(reference_resolution: int, comparison_resolution: int, plot=False, single=False):
    reference_dir = Path(f"/home/lukas/monofonic_tests/shannon_{reference_resolution}_100")
    comparison_dir = Path(f"/home/lukas/monofonic_tests/shannon_{comparison_resolution}_100/")

    ref_masses = []
    comp_masses = []
    ref_sizes = []
    comp_sizes = []
    matches = []

    print("reading reference file")
    df_ref, ref_meta = read_file(reference_dir)
    df_ref_halo = read_halo_file(reference_dir)

    print("reading comparison file")
    df_comp, comp_meta = read_file(comparison_dir)
    df_comp_halo = read_halo_file(comparison_dir)

    print(f"Memory ref: {memory_usage(df_ref):.2f} MB")
    print(f"Memory comp: {memory_usage(df_comp):.2f} MB")

    for index, original_halo in df_ref_halo.iterrows():
        print(f"{index} of {len(df_ref_halo)} original halos")
        particles_in_ref_halo = df_ref.loc[df_ref["FOFGroupIDs"] == index]
        ref_halo = df_ref_halo.loc[index]
        # cumulative_mass_profile(particles_in_ref_halo, ref_halo, ref_meta, plot=plot)
        halo_particle_ids = set(particles_in_ref_halo.index.to_list())

        prev_len = len(halo_particle_ids)
        if reference_resolution < comparison_resolution:
            print("upscaling IDs")
            upscaled_ids = set()
            scaler = IDScaler(reference_resolution, comparison_resolution)
            # i = 0
            for id in halo_particle_ids:
                # i += 1
                # if i % 1000 == 0:
                #     print(i)
                upscaled_ids.update(set(scaler.upscale(id)))
            halo_particle_ids = upscaled_ids
            after_len = len(upscaled_ids)
            print(f"{prev_len} => {after_len} (factor {after_len / prev_len})")
        if comparison_resolution < reference_resolution:
            print("downscaling IDs")
            downscaled_ids = set()
            scaler = IDScaler(comparison_resolution, reference_resolution)
            for id in halo_particle_ids:
                downscaled_ids.add(scaler.downscale(id))
            halo_particle_ids = downscaled_ids
            print("done")
            after_len = len(halo_particle_ids)
            print(f"{prev_len} => {after_len} (factor {prev_len / after_len})")

        print("look up halo particles in comparison dataset")
        particles = df_comp.loc[list(halo_particle_ids)]

        halos_in_particles = set(particles["FOFGroupIDs"])
        halos_in_particles.discard(2147483647)
        print(f"{len(halos_in_particles)} halos found in new particles")
        if plot:
            fig: Figure = plt.figure()
            ax: Axes = fig.gca()
            ax.scatter(particles["X"], particles["Y"], s=1, alpha=.3, label="Halo")
        #     ax.scatter(particles_in_ref_halo["X"], particles_in_ref_halo["Y"], s=1, alpha=.3, label="RefHalo")
        # plt.legend()
        # plt.show()
        best_halo = None
        best_halo_match = 0

        for i, halo in enumerate(halos_in_particles):
            # print("----------", halo, "----------")
            print_progress(i, len(halos_in_particles), halo)
            # halo_data = df_comp_halo.loc[halo]
            particles_in_comp_halo: DataFrame = df_comp.loc[df_comp["FOFGroupIDs"] == halo]
            halo_size = len(particles_in_comp_halo)

            df = particles_in_comp_halo.join(particles, how="inner", rsuffix="ref")
            shared_size = len(df)
            match = shared_size / halo_size
            # print(match, halo_size, shared_size)
            # print(df)
            if plot:
                ax.scatter(df["X"], df["Y"], s=1, alpha=.3, label=f"shared {halo}")
            # ax.scatter(particles_in_comp_halo["X"], particles_in_comp_halo["Y"], s=2, alpha=.3, label=f"shared {halo}")
            if shared_size > best_halo_match:
                best_halo_match = shared_size
                best_halo = halo

        # print("-------")
        # print(best_halo)
        comp_halo = df_comp_halo.loc[best_halo]

        print(ref_halo)
        print(comp_halo)
        ref_sizes.append(ref_halo["Sizes"])
        ref_masses.append(ref_halo["Masses"])
        comp_sizes.append(comp_halo["Sizes"])
        comp_masses.append(comp_halo["Masses"])
        matches.append(best_halo_match / len(particles))
        # exit()
        if plot:
            ax.legend()
            ax.set_title(f"{reference_dir.name} vs. {comparison_dir.name} (Halo {index})")
            fig.savefig("out.png", dpi=300)
            plt.show()
        if single:
            break

    df = DataFrame(np.array([matches, ref_sizes, comp_sizes, ref_masses, comp_masses]).T,
                   columns=["matches", "ref_sizes", "comp_sizes", "ref_masses", "comp_masses"])
    print(df)
    df.to_csv("sizes.csv", index=False)
    return df, reference_dir.name + "_" + comparison_dir.name


if __name__ == '__main__':
    compare_halo_resolutions(
        reference_resolution=128,
        comparison_resolution=512,
        plot=False,
        single=False
    )
