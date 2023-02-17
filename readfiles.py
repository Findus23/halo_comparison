import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class ParticlesMeta:
    particle_mass: float


def read_file(file: Path) -> Tuple[pd.DataFrame, ParticlesMeta]:
    cache_file = file.with_suffix(".cache.pickle")
    meta_cache_file = file.with_suffix(".cache_meta.pickle")
    cache_is_outdated = file.stat().st_mtime > cache_file.stat().st_mtime
    if cache_is_outdated:
        print("cache is outdated")
    if not (cache_file.exists() and meta_cache_file.exists()) or cache_is_outdated:
        reference_file = h5py.File(file)
        has_fof = "FOFGroupIDs" in reference_file["PartType1"]

        try:
            masses = reference_file["PartType1"]["Masses"]
            if not np.all(masses == masses[0]):
                raise ValueError("only equal mass particles are supported for now")
            meta = ParticlesMeta(particle_mass=masses[0])

        except KeyError:
            meta = ParticlesMeta(particle_mass=0)
        df = pd.DataFrame(
            reference_file["PartType1"]["Coordinates"], columns=["X", "Y", "Z"]
        )
        if has_fof:
            df2 = pd.DataFrame(
                reference_file["PartType1"]["FOFGroupIDs"], columns=["FOFGroupIDs"]
            ).astype("category")
            df = df.merge(df2, "outer", left_index=True, right_index=True)
            del df2
        df3 = pd.DataFrame(
            reference_file["PartType1"]["ParticleIDs"], columns=["ParticleIDs"]
        )

        df = df.merge(df3, "outer", left_index=True, right_index=True)
        del df3
        df.set_index("ParticleIDs", inplace=True)
        if has_fof:
            print("sorting")
            df.sort_values("FOFGroupIDs", inplace=True)
        print("saving cache")
        with meta_cache_file.open("wb") as f:
            pickle.dump(meta, f)
        df.to_pickle(str(cache_file))
        reference_file.close()
        return df, meta
    print("from cache")
    df = pd.read_pickle(str(cache_file))
    with meta_cache_file.open("rb") as f:
        meta = pickle.load(f)
    return df, meta


def read_halo_file(file: Path) -> DataFrame:
    # file = path / "fof_output_0004.hdf5"
    reference_file = h5py.File(file)
    df1 = pd.DataFrame(reference_file["Groups"]["Centres"], columns=["X", "Y", "Z"])
    df2 = pd.DataFrame(reference_file["Groups"]["GroupIDs"], columns=["GroupIDs"])
    df3 = pd.DataFrame(reference_file["Groups"]["Masses"], columns=["Masses"])
    df4 = pd.DataFrame(reference_file["Groups"]["Sizes"], columns=["Sizes"])
    df = df1.merge(df2, "outer", left_index=True, right_index=True)
    df = df.merge(df3, "outer", left_index=True, right_index=True)
    df = df.merge(df4, "outer", left_index=True, right_index=True)
    df.set_index("GroupIDs", inplace=True)
    return df


def read_gadget_halos(directory: Path):
    reference_file = h5py.File(directory / "output" / "fof_subhalo_tab_019.hdf5")
    df1 = pd.DataFrame(reference_file["Subhalo"]["SubhaloPos"], columns=["X", "Y", "Z"])
    df2 = pd.DataFrame(reference_file["Subhalo"]["SubhaloMass"], columns=["Masses"])
    df = df1.merge(df2, "outer", left_index=True, right_index=True)
    return df


def read_fof_file(path: Path):
    file = path / ""


def read_g4_file(file: Path, zoom_type: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    with h5py.File(file) as reference_file:

        hubble_param = reference_file["Parameters"].attrs["HubbleParam"]
        masstable = reference_file['Header'].attrs['MassTable']
        if zoom_type == 'pbh':
            highres_parttype = 'PartType0'
            lowres_parttype = 'PartType1'
            highres_mass = masstable[0] / hubble_param
            lowres_mass = masstable[1] / hubble_param

        elif zoom_type == 'cdm':
            highres_parttype = 'PartType1'
            lowres_parttype = 'PartType2'
            highres_mass = masstable[1] / hubble_param
            lowres_mass = masstable[2] / hubble_param
        else:
            raise ValueError('Please select pbh or cdm as zoom_type!')

        # all coordinates in Mpc/h without adaption!
        highres_coordinates = reference_file[highres_parttype]["Coordinates"][:]
        lowres_coordinates = reference_file[lowres_parttype]["Coordinates"][:]

    return highres_coordinates, lowres_coordinates, highres_mass, lowres_mass
