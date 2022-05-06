from pathlib import Path

import numpy as np
import pandas as pd
import pyvista

from readfiles import read_file


def filter_for_3d(df: pd.DataFrame, group: int):
    df_ref = df.loc[df["FOFGroupIDs"] == group]

    xs = df_ref.X
    ys = df_ref.Y
    zs = df_ref.Z
    return np.array([xs, ys, zs]).T


reference_dir = Path(f"/home/lukas/monofonic_tests/shannon_512_100")
df_ref, _ = read_file(reference_dir)

data = filter_for_3d(df_ref, group=1)
data2 = filter_for_3d(df_ref, group=2)
data3 = filter_for_3d(df_ref, group=3)
print(len(data))

for data in [filter_for_3d(df_ref, group=1),filter_for_3d(df_ref, group=2),filter_for_3d(df_ref, group=3)]:

    pdata = pyvista.PointSet(data)
    pdata.plot(point_size=3, render_points_as_spheres=True, eye_dome_lighting=True, parallel_projection=True,
               anti_aliasing=True)
