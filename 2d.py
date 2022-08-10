"""
Looks like this file takes one snapshot (line 26) and visualises specified halos (line 28) as a 2d-projection. Might not be accurate, hasn't been changed in a long time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from paths import base_dir
from readfiles import read_file


def filter_for_2d(df: pd.DataFrame, group: int):
    xs = df_ref.X
    ys = df_ref.Y
    return np.array([xs, ys]).T


reference_dir = Path(base_dir / f"shannon_512_100")
df_ref, _ = read_file(reference_dir / "output_0004.hdf5")

df = df_ref.loc[df_ref["FOFGroupIDs"] == 1]
df2 = df_ref.loc[df_ref["FOFGroupIDs"] == 4]

fig: Figure = plt.figure()
ax: Axes = fig.gca()

# ax.hist2d(df.X, df.Y, bins=500, norm=LogNorm())
ax.hist2d(df2.X, df2.Y, bins=1000, norm=LogNorm())

plt.show()
