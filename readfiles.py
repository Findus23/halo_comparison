from pathlib import Path

import h5py
import pandas as pd
from pandas import DataFrame


def read_file(path: Path) -> pd.DataFrame:
    cache_file = path / "cache"
    if not cache_file.exists():
        file = path / "output_0004.hdf5"
        reference_file = h5py.File(file)
        df = pd.DataFrame(reference_file["PartType1"]["Coordinates"], columns=["X", "Y", "Z"])
        df2 = pd.DataFrame(reference_file["PartType1"]["FOFGroupIDs"], columns=["FOFGroupIDs"]).astype("category")
        df = df.merge(df2, "outer", left_index=True, right_index=True)
        del df2
        df3 = pd.DataFrame(reference_file["PartType1"]["ParticleIDs"], columns=["ParticleIDs"])

        df = df.merge(df3, "outer", left_index=True, right_index=True)
        del df3
        df.set_index("ParticleIDs", inplace=True)
        print("saving cache")
        df.to_pickle(str(cache_file))
        return df
    print("from cache")
    df = pd.read_pickle(str(cache_file))
    return df


def read_halo_file(path: Path) -> DataFrame:
    file = path / "fof_output_0004.hdf5"
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
