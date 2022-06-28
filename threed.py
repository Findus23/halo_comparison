import sys
from pathlib import Path

import pandas as pd
import pyvista
import vtk
from pandas import DataFrame
from pyvista import Plotter

from paths import base_dir
from readfiles import read_file, read_halo_file

pyvista.set_plot_theme("dark")  # "paraview"


def plotdf3d(pl: Plotter, df: DataFrame, color="white"):
    data = df_to_coords(df)
    pdata = pyvista.PointSet(data[::, 0:3])
    # pdata.plot(point_size=1, scalars=data[::, -1], render_points_as_spheres=False, parallel_projection=True,
    #            anti_aliasing=True, opacity=0.2)
    # pl.add_points(a) would be equivalent to pl.add_mesh(style="points")
    pl.add_mesh(
        pdata,
        point_size=1,
        style="points",
        opacity=0.1,
        color=color,
        # scalars=data[::, -1],
    )
    # pl.camera_position
    # pl.enable_stereo_render()
    # pl.ren_win.SetStereoTypeToSplitViewportHorizontal()
    # pl.show_grid()
    pl.enable_terrain_style()
    pl.enable_parallel_projection()
    pl.camera.zoom(2)
    renderer = pl.renderer
    basic_passes = vtk.vtkRenderStepsPass()
    blur_pass = vtk.vtkGaussianBlurPass()
    blur_pass.SetDelegatePass(basic_passes)

    glrenderer = vtk.vtkOpenGLRenderer.SafeDownCast(renderer)
    glrenderer.SetPass(blur_pass)

def df_to_coords(df: pd.DataFrame):
    return df[["X", "Y", "Z"]].to_numpy()


if __name__ == '__main__':
    # HALO = 1
    # reference_dir = base_dir / "shannon_512_100"
    # df, _ = read_file(reference_dir / "output_0004.hdf5")
    # df_halo = read_halo_file(reference_dir / "fof_output_0004.hdf5").loc[HALO]
    # df = df.loc[df["FOFGroupIDs"] == HALO]
    # df.to_csv("halo.csv")
    input_file = Path(sys.argv[1])
    df, _ = read_file(input_file)
    pl = Plotter()
    plotdf3d(pl, df)
    # pl.set_focus((df_halo.X,df_halo.Y,df_halo.Z))
    pl.show()
