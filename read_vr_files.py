from pathlib import Path
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
from h5py import Dataset
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from halo_mass_profile import V
from paths import base_dir
from readfiles import read_file
from utils import print_progress

HaloParticleMapping = Dict[int, set[int]]


def all_children(df: DataFrame, id: int):
    subhalos: pd.DataFrame = df.loc[df.parent_halo_id == id]

    if len(subhalos) == 0:
        yield id

    for sh, value in subhalos.iterrows():
        yield from all_children(df, sh)


def particles_in_halo(
    offsets: Dict[int, int], particle_ids: np.ndarray
) -> HaloParticleMapping:
    """
    get mapping from halo ID to particle ID set by using the offset and a lookup in the particle array
    """
    halo_particle_ids: Dict[int, set[int]] = {}
    pointer = 0
    for halo_id, offset in offsets.items():
        if halo_id % 100 == 0:
            print_progress(halo_id, len(offsets))
        if halo_id != len(offsets):
            end = offsets[halo_id + 1]
        else:  # special handling for last halo
            end = -1
        ids = particle_ids[pointer:end]
        halo_particle_ids[halo_id] = set(ids)
        pointer = end
    return halo_particle_ids
    # if recursivly:
    #     # add particles of subhalos to parent halos
    #     # maybe not such a good idea
    #     for halo_id in range(1, len(df) + 1):
    #         sub_n_children = list(all_children(df, halo_id))  # IDs of children, subchildren, ...
    #         if not sub_n_children:
    #             continue
    #         for child_id in sub_n_children:
    #             child_particles = halo_particle_ids[child_id]
    #             halo_particle_ids[halo_id].update(child_particles)


def cached_particles_in_halo(file: Path, *args, **kwargs) -> HaloParticleMapping:
    """
    Save mapping from halo ID to set of particle IDs into HDF5 file.
    Every halo is a dataset with its ID as the name and the list of particles as value.

    Unfortunatly this is magnitudes slower than doing the calculation itself as HDF5 is not
    intended for 100K small datasets making this whole function pointless.

    """
    if file.exists():
        print("loading from cache")
        with h5py.File(file) as data_file:
            halo_particle_ids: HaloParticleMapping = {}
            i = 0
            for name, dataset in data_file.items():
                halo_particle_ids[int(name)] = set(dataset)
                print_progress(i, len(data_file))
                i += 1
    else:
        halo_particle_ids = particles_in_halo(*args, **kwargs)
        print("saving to cache")
        with h5py.File(file, "w") as data_file:
            for key, valueset in halo_particle_ids.items():
                data_file.create_dataset(
                    str(key),
                    data=list(valueset),
                    compression="gzip",
                    compression_opts=5,
                )
    return halo_particle_ids


def read_velo_halos(directory: Path, veloname="vroutput"):
    """
    Returns a dataframe containing all scalar properties of the halos
    (https://velociraptor-stf.readthedocs.io/en/latest/output.html),
    """
    if (directory / f"{veloname}.catalog_groups.0").exists():
        suffix = ".0"
    else:
        suffix = ""
    group_catalog = h5py.File(directory / f"{veloname}.catalog_groups{suffix}")
    group_properties = h5py.File(directory / f"{veloname}.properties{suffix}")
    scalar_properties = {}
    for k, v in group_properties.items():
        if not isinstance(v, Dataset):
            # skip groups
            continue
        if len(v.shape) != 1:
            print(v)
            continue
        if len(v) == 1:
            # skip global properties like Total_num_of_groups
            continue
        scalar_properties[k] = v
    scalar_properties["X"] = scalar_properties["Xc"]
    scalar_properties["Y"] = scalar_properties["Yc"]
    scalar_properties["Z"] = scalar_properties["Zc"]
    data = {
        "group_size": group_catalog["Group_Size"],
        "offset": group_catalog["Offset"],
        "offset_unbound": group_catalog["Offset_unbound"],
        "parent_halo_id": group_catalog["Parent_halo_ID"],
    }
    df = pd.DataFrame(
        {**data, **scalar_properties}
    )  # create dataframe from two merged dicts
    df.index += 1  # Halo IDs start at 1
    return df


def read_velo_halo_particles(
    directory: Path, skip_halo_particle_ids=False, skip_unbound=True
) -> Tuple[DataFrame, Optional[HaloParticleMapping], Optional[HaloParticleMapping]]:
    """
    This reads the output files of VELOCIraptor
    and returns the halo data from read_velo_halos
    and two dictionaries mapping the halo IDs to sets of particle IDs
    """
    if (directory / f"vroutput.catalog_particles.0").exists():
        suffix = ".0"
    else:
        suffix = ""
    df = read_velo_halos(directory)
    particle_catalog = h5py.File(directory / f"vroutput.catalog_particles{suffix}")
    particle_ids = np.asarray(particle_catalog["Particle_IDs"])

    particle_catalog_unbound = h5py.File(
        directory / f"vroutput.catalog_particles.unbound{suffix}"
    )
    particle_ids_unbound = particle_catalog_unbound["Particle_IDs"][:]
    if skip_halo_particle_ids:
        return df, None, None
    print("look up bound particle IDs")
    # particle_cache_file = directory / "vrbound.cache.hdf5"
    # ub_particle_cache_file = directory / "vrunbound.cache.hdf5"
    halo_offsets = dict(df["offset"])
    halo_particle_ids = particles_in_halo(halo_offsets, particle_ids)
    if skip_unbound:
        halo_particle_unbound_ids = {}
    else:
        print("look up unbound particle IDs")
        halo_unbound_offsets = dict(df["offset_unbound"])
        halo_particle_unbound_ids = particles_in_halo(
            halo_unbound_offsets, particle_ids_unbound
        )
    return df, halo_particle_ids, halo_particle_unbound_ids


def read_velo_profiles(directory: Path):
    if (directory / f"vroutput.profiles.0").exists():
        suffix = ".0"
    else:
        suffix = ""
    profiles_file = h5py.File(directory / f"vroutput.profiles{suffix}")
    bin_edges = profiles_file["Radial_bin_edges"][1:]

    bin_volumes = V(bin_edges)

    mass_profiles = profiles_file["Mass_profile"][:]

    density_profiles = mass_profiles / bin_volumes

    return bin_edges, mass_profiles, density_profiles


def main():
    waveform = "shannon"
    Nres = 512
    directory = base_dir / f"{waveform}_{Nres}_100"

    df_halo, halo_particle_ids, halo_particle_unbound_ids = read_velo_halo_particles(
        directory
    )
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


if __name__ == "__main__":
    main()
