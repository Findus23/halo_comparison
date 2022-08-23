from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from temperatures import calculate_T
from utils import create_figure


def filter_3d(
        coords: np.ndarray, data: np.ndarray,
        extent: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    filter = (
            (extent[0] < coords[::, 0]) &
            (coords[::, 0] < extent[1]) &

            (extent[2] < coords[::, 1]) &
            (coords[::, 1] < extent[3])
    )
    print("before", coords.shape)
    data = data[filter]
    coords = coords[filter]

    print("after", coords.shape)
    return coords, data


def create_2d_slice(
        input_file: Path, center: List[float], extent,
        property="InternalEnergies", method="nearest"
) -> np.ndarray:
    cut_axis = 2  # Z
    with h5py.File(input_file) as f:
        pt0 = f["PartType0"]
        coords = pt0["Coordinates"][:]
        data = pt0[property if property != "Temperatures" else "InternalEnergies"][:]

        coords, data = filter_3d(coords, data, extent)
        if property == "Temperatures":
            print("calculating temperatures")
            data = np.array([calculate_T(u) for u in data])

        xrange = np.linspace(extent[0],extent[1], 1000)
        yrange = np.linspace(extent[2],extent[3], 1000)
        gx, gy, gz = np.meshgrid(xrange, yrange, center[cut_axis])
        print("interpolating")
        grid = griddata(coords, data, (gx, gy, gz), method=method)[::, ::, 0]
        return grid
        print(grid.shape)
        # stats, x_edge, y_edge, _ = binned_statistic_2d(
        #     coords_in_slice[::, x_axis],
        #     coords_in_slice[::, y_axis],
        #     data_in_slice,
        #     bins=500,
        #     statistic="mean"
        # )
        fig, ax = create_figure()
        # stats = np.nan_to_num(stats)
        print("plotting")
        img = ax.imshow(
            grid,
            norm=LogNorm(),
            interpolation="nearest",
            origin="lower",
            extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]],
        )
        ax.set_title(input_file.parent.stem)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_aspect("equal")
        fig.colorbar(img, label="Temperatures")
        fig.tight_layout()
        plt.show()
