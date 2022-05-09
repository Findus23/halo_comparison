from pathlib import Path

import h5py
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from paths import base_dir

from readfiles import read_file


def all_children(df, id):
    subhalos: pd.DataFrame = df.loc[df.parent_halo_id == id]

    if len(subhalos) == 0:
        yield id

    for sh, value in subhalos.iterrows():
        yield from all_children(df, sh)


def subhalo_particles_recursively(halo: int, waveform: str, Nres: int):
    directory = base_dir / f"{waveform}_{Nres}_100"

    group_catalog = h5py.File(directory / "vroutput.catalog_groups")
    df = pd.DataFrame(
        {
            "group_size": group_catalog["Group_Size"],
            "offset": group_catalog["Offset"],
            "offset_unbound": group_catalog["Offset_unbound"],
            "parent_halo_id": group_catalog["Parent_halo_ID"],
        }
    )

    particle_catalog = h5py.File(directory / "vroutput.catalog_particles")
    particle_ids = particle_catalog["Particle_IDs"][:]

    particle_catalog_unbound = h5py.File(
        directory / "vroutput.catalog_particles.unbound"
    )
    particle_ids_unbound = particle_catalog_unbound["Particle_IDs"][:]

    all_halo_ids = [halo] + list(all_children(df, halo))

    all_halo_offsets = [[df["offset"][j], df["offset"][j + 1]] for j in all_halo_ids]
    all_halo_offsets_unbound = [
        [df["offset_unbound"][j], df["offset_unbound"][j + 1]] for j in all_halo_ids
    ]

    all_halo_particles = [
        set(particle_ids[all_halo_offsets[j][0] : all_halo_offsets[j][1]])
        for j in range(len(all_halo_offsets))
    ]
    all_halo_particles_unbound = [
        set(
            particle_ids_unbound[
                all_halo_offsets_unbound[j][0] : all_halo_offsets_unbound[j][1]
            ]
        )
        for j in range(len(all_halo_offsets_unbound))
    ]

    return all_halo_ids, all_halo_particles, all_halo_particles_unbound


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

