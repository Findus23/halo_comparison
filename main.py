from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from cumulative_mass_profiles import cumulative_mass_profile
from readfiles import read_file, read_halo_file
from remap_particle_IDs import IDScaler

REFERENCE_RESOLUTION = 128
COMPARISON_RESOLUTION = 128
PLOT = True
SINGLE = False

reference_dir = Path(f"/home/lukas/monofonic_tests/shannon_{REFERENCE_RESOLUTION}_100")
comparison_dir = Path(f"/home/lukas/monofonic_tests/DB8_{COMPARISON_RESOLUTION}_100/")

ref_masses = []
comp_masses = []
ref_sizes = []
comp_sizes = []

print("reading reference file")
df_ref = read_file(reference_dir)
df_ref_halo = read_halo_file(reference_dir)

print("reading comparison file")
df_comp = read_file(comparison_dir)
df_comp_halo = read_halo_file(comparison_dir)

bytes_used = df_ref.memory_usage(index=True).sum()
print(f"Memory: {bytes_used / 1024 / 1024:.2f} MB")
print(df_ref.dtypes)

for index, original_halo in df_ref_halo[:5].iterrows():
    print(index)
    print(len(df_ref))
    particles_in_ref_halo = df_ref.loc[df_ref["FOFGroupIDs"] == index]
    ref_halo = df_ref_halo.loc[index]
    print("halo", ref_halo,len(particles_in_ref_halo))
    cumulative_mass_profile(particles_in_ref_halo,ref_halo)
    halo_particle_ids = set(particles_in_ref_halo.index.to_list())

    if REFERENCE_RESOLUTION < COMPARISON_RESOLUTION:
        print("upscaling IDs")
        upscaled_ids = set()
        prev_len = len(halo_particle_ids)
        print(prev_len)
        scaler = IDScaler(REFERENCE_RESOLUTION, COMPARISON_RESOLUTION)
        # i = 0
        for id in halo_particle_ids:
            # i += 1
            # if i % 1000 == 0:
            #     print(i)
            upscaled_ids.update(set(scaler.upscale(id)))
        halo_particle_ids = upscaled_ids
        after_len = len(upscaled_ids)
        print(after_len)
        print(after_len / prev_len)
        print("done")
    if COMPARISON_RESOLUTION < REFERENCE_RESOLUTION:
        print("downscaling IDs")
        prev_count = len(halo_particle_ids)
        print(prev_count)
        downscaled_ids = set()
        scaler = IDScaler(COMPARISON_RESOLUTION, REFERENCE_RESOLUTION)
        for id in halo_particle_ids:
            downscaled_ids.add(scaler.downscale(id))
        halo_particle_ids = downscaled_ids
        print("done")
        after_count = len(halo_particle_ids)
        print(after_count)
        print(prev_count / after_count)

    particles = df_comp.loc[list(halo_particle_ids)]
    # print(particles)

    halos_in_particles = set(particles["FOFGroupIDs"])
    halos_in_particles.discard(2147483647)
    # print(halos_in_particles)
    if PLOT:
        fig: Figure = plt.figure()
        ax: Axes = fig.gca()
        ax.scatter(particles["X"], particles["Y"], s=1, alpha=.3, label="Halo")
    #     ax.scatter(particles_in_ref_halo["X"], particles_in_ref_halo["Y"], s=1, alpha=.3, label="RefHalo")
    # plt.legend()
    # plt.show()
    best_halo = None
    best_halo_match = 0

    for halo in halos_in_particles:
        # print("----------", halo, "----------")
        print(halo)
        halo_data = df_comp_halo.loc[halo]
        particles_in_comp_halo: DataFrame = df_comp.loc[df_comp["FOFGroupIDs"] == halo]
        halo_size = len(particles_in_comp_halo)

        df = particles_in_comp_halo.join(particles, how="inner", rsuffix="ref")
        shared_size = len(df)
        match = shared_size / halo_size
        # print(match, halo_size, shared_size)
        # print(df)
        if PLOT:
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
    # exit()
    if PLOT:
        ax.legend()
        ax.set_title(f"{reference_dir.name} vs. {comparison_dir.name} (Halo {index})")
        fig.savefig("out.png", dpi=300)
        plt.show()
    if SINGLE:
        break

df = DataFrame(np.array([ref_sizes, comp_sizes, ref_masses, comp_masses]).T,
               columns=["ref_sizes", "comp_sizes", "ref_masses", "comp_masses"])
print(df)
df.to_csv("sizes.csv", index=False)
