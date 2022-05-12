from pathlib import Path
from typing import Dict

import h5py
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paths import base_dir
from readfiles import read_file
from utils import print_progress


def all_children(df, id):
    subhalos: pd.DataFrame = df.loc[df.parent_halo_id == id]

    if len(subhalos) == 0:
        yield id

    for sh, value in subhalos.iterrows():
        yield from all_children(df, sh)


def particles_in_halo(df, particle_ids, recursivly: bool, unbound: bool):
    halo_particle_ids: Dict[int, set[int]] = {}
    pointer = 0
    for halo_id in range(1, len(df) + 1):
        halo = df.loc[halo_id]
        print_progress(halo_id, len(df), halo.group_size)
        if halo_id != len(df):
            next_halo = df.loc[halo_id + 1]
            end = int(next_halo["offset_unbound" if unbound else "offset"])
        else:  # special handling for last halo
            end = -1
        particles_in_halo = particle_ids[pointer:end]
        ids = set(particles_in_halo)
        halo_particle_ids[halo_id] = ids
        pointer = end
    if recursivly:
        for halo_id in range(1, len(df) + 1):
            sub_n_children = list(all_children(df, halo_id))  # IDs of children, subchildren, ...
            if not sub_n_children:
                continue
            for child_id in sub_n_children:
                child_particles = halo_particle_ids[child_id]
                halo_particle_ids[halo_id].update(child_particles)
    return halo_particle_ids


def read_velo_halos(directory: Path, recursivly=True, skip_unbound=False):
    group_catalog = h5py.File(directory / "vroutput.catalog_groups")
    group_properties = h5py.File(directory / "vroutput.properties")
    df = pd.DataFrame(
        {
            "group_size": group_catalog["Group_Size"],
            "offset": group_catalog["Offset"],
            "offset_unbound": group_catalog["Offset_unbound"],
            "parent_halo_id": group_catalog["Parent_halo_ID"],
            "X": group_properties["Xc"],
            "Y": group_properties["Yc"],
            "Z": group_properties["Zc"],
            "Rvir": group_properties["Rvir"],
            "Mass_tot": group_properties["Mass_200crit"]
        }
    )
    df.index += 1  # set Halo IDs start at 1

    particle_catalog = h5py.File(directory / "vroutput.catalog_particles")
    particle_ids = particle_catalog["Particle_IDs"][:]

    particle_catalog_unbound = h5py.File(
        directory / "vroutput.catalog_particles.unbound"
    )
    particle_ids_unbound = particle_catalog_unbound["Particle_IDs"][:]

    print("look up bound particle IDs")
    halo_particle_ids = particles_in_halo(df, particle_ids, recursivly, unbound=False)
    if skip_unbound:
        halo_particle_unbound_ids = {}
    else:
        print("look up unbound particle IDs")
        halo_particle_unbound_ids = particles_in_halo(df, particle_ids_unbound, recursivly, unbound=True)

    return df, halo_particle_ids, halo_particle_unbound_ids


def main():
    waveform = "shannon"
    Nres = 512
    directory = base_dir / f"{waveform}_{Nres}_100"

    df_halo, halo_particle_ids, halo_particle_unbound_ids = read_velo_halos(directory, recursivly=True)
    particles, meta = read_file(directory)
    HALO = 1000
    while True:
        fig: Figure = plt.figure()
        ax: Axes = fig.gca()

        bound_particles = particles.loc[list(halo_particle_ids[HALO])]
        unbound_particles = particles.loc[list(halo_particle_unbound_ids[HALO])]

        ax.scatter(bound_particles.X, bound_particles.Y, label="bound", s=1)
        ax.scatter(unbound_particles.X, unbound_particles.Y, label="unbound", s=1)

        plt.show()
        HALO += 1


if __name__ == '__main__':
    main()

# #This could be used to make a 2-D plot, but that doesn't teach us much about the halos:
# particles, _ = read_file(directory)
# print(particles)
# print(particle_ids_in_halo)
# bound_particles = particles.loc[list(particle_ids_in_halo)]
# unbound_particles = particles.loc[list(particle_ids_unbound_in_halo)]

# plt.scatter(bound_particles['X'], bound_particles['Z'], label='bound', s=1)
# plt.scatter(unbound_particles['X'], unbound_particles['Z'], label='unbound', s=1)
# plt.legend()
# plt.show()


# #This was an initial test to see if reading in works as expected
# lower_offset = df['offset'][test_group]
# higher_offset = df['offset'][test_group + 1]
# particle_ids_in_halo = set(particle_ids[lower_offset:higher_offset])

# lower_offset_unbound = df['offset_unbound'][test_group]
# higher_offset_unbound = df['offset_unbound'][test_group + 1]
# particle_ids_unbound_in_halo = set(particle_ids_unbound[lower_offset_unbound:higher_offset_unbound])

# size = df['group_size'][test_group]

# assert size == len(particle_ids_in_halo) + len(particle_ids_unbound_in_halo)
