import sys
from pathlib import Path
import numpy as np

import pyvista
import vtk
from pandas import DataFrame
from pyvista import Plotter

from paths import cdm_zoom_dir, pbh_zoom_dir
from readfiles import read_g4_file

pyvista.set_plot_theme("dark")  # "paraview"

def plotnp3d(pl: Plotter, data: np.ndarray, color="white"):
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

if __name__ == "__main__":
    input_file = cdm_zoom_dir / f"snapshot_{int(sys.argv[1]):03d}.hdf5"
    highres_coords, lowres_coords = read_g4_file(input_file, 'cdm')

    random_highres_selection = highres_coords[np.random.choice(highres_coords.shape[0], 100000, replace=False)]
    random_lowres_selection = lowres_coords[np.random.choice(lowres_coords.shape[0], 100000, replace=False)]

    pl = Plotter()
    plotnp3d(pl, random_lowres_selection)
    plotnp3d(pl, random_highres_selection)
    pl.show()
