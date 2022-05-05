from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.cpu.pybind.visualization import RenderOption

from readfiles import read_file


def file_to_arry(file: Path):
    df_ref = read_file(file)
    df_ref = df_ref.loc[df_ref["FOFGroupIDs"] == 34]

    xs = df_ref.X
    ys = df_ref.Y
    zs = df_ref.Z
    print()
    return np.array([xs, ys, zs]).T


reference_dir = Path(f"/home/lukas/monofonic_tests/shannon_512_100")
data = file_to_arry(reference_dir)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(data2)

# downpcd = pcd.voxel_down_sample(voxel_size=5)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=1, max_nn=300))
# o3d.visualization.draw_geometries([downpcd])
# pcd.colors = o3d.utility.Vector3dVector(colors / 65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([pcd])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
opt: RenderOption = vis.get_render_option()
opt.point_size=5

vis.run()
vis.destroy_window()
