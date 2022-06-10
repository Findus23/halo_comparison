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
    if not (cache_file.exists() and meta_cache_file.exists()):
        reference_file = h5py.File(file)
        has_fof = "FOFGroupIDs" in reference_file["PartType1"]

        masses = reference_file["PartType1"]["Masses"]
        if not np.all(masses == masses[0]):
            raise ValueError("only equal mass particles are supported for now")
        df = pd.DataFrame(reference_file["PartType1"]["Coordinates"], columns=["X", "Y", "Z"])
        if has_fof:
            df2 = pd.DataFrame(reference_file["PartType1"]["FOFGroupIDs"], columns=["FOFGroupIDs"]).astype("category")
            df = df.merge(df2, "outer", left_index=True, right_index=True)
            del df2
        df3 = pd.DataFrame(reference_file["PartType1"]["ParticleIDs"], columns=["ParticleIDs"])

        df = df.merge(df3, "outer", left_index=True, right_index=True)
        del df3
        df.set_index("ParticleIDs", inplace=True)
        if has_fof:
            print("sorting")
            df.sort_values("FOFGroupIDs", inplace=True)
        meta = ParticlesMeta(
            particle_mass=masses[0]
        )
        print("saving cache")
        with meta_cache_file.open("wb") as f:
            pickle.dump(meta, f)
        df.to_pickle(str(cache_file))
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


def read_fof_file(path: Path):
    file = path / ""
