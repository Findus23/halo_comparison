import numpy as np
import pandas as pd
import pyvista
from pyvista import themes

from paths import base_dir
from readfiles import read_file


def filter_for_3d(df: pd.DataFrame, group: int):
    df_ref = df.loc[df["FOFGroupIDs"] == group]

    xs = df_ref.X
    ys = df_ref.Y
    zs = df_ref.Z
    data = df_ref.index
    return np.array([xs, ys, zs, data]).T


reference_dir = base_dir / "shannon_512_100"
df_ref, _ = read_file(reference_dir)
pyvista.set_plot_theme(themes.DarkTheme())
pl = pyvista.Plotter()
for data in [filter_for_3d(df_ref, group=1), filter_for_3d(df_ref, group=2), filter_for_3d(df_ref, group=3)]:
    pdata = pyvista.PointSet(data[::, 0:3])
    # pdata.plot(point_size=1, scalars=data[::, -1], render_points_as_spheres=False, parallel_projection=True,
    #            anti_aliasing=True, opacity=0.2)
    # pl.add_points(a) would be equivalent to pl.add_mesh(style="points")
    pl.add_mesh(
        pdata,
        point_size=1,
        style="points",
        opacity=0.1,
        scalars=data[::, -1],
    )

    # pl.enable_stereo_render()
    pl.show_grid()
    pl.enable_parallel_projection()
    pl.show()
    exit()
