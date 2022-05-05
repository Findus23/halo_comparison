from pathlib import Path

import numpy as np
import pyvista

from readfiles import read_file


def file_to_arry(file: Path):
    df_ref = read_file(file)
    df_ref = df_ref.loc[df_ref["FOFGroupIDs"] == 1]

    xs = df_ref.X
    ys = df_ref.Y
    zs = df_ref.Z
    print()
    return np.array([xs, ys, zs]).T


reference_dir = Path(f"/home/lukas/monofonic_tests/shannon_512_100")

data = file_to_arry(reference_dir)
print(len(data))
# point_cloud = np.random.random((100, 3))
pdata = pyvista.PointSet(data)
pdata.plot(point_size=3, render_points_as_spheres=True, eye_dome_lighting=True, parallel_projection=True,
           anti_aliasing=True)
# # create many spheres from the point cloud
# pc.plot(cmap='Reds')
