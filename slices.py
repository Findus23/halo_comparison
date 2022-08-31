from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from temperatures import calculate_T
from utils import create_figure


def filter_3d(
        coords: np.ndarray, extent: List[float], data: np.ndarray = None, zlimit=None

) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    filter = (
            (extent[0] < coords[::, 0]) &
            (coords[::, 0] < extent[1]) &

            (extent[2] < coords[::, 1]) &
            (coords[::, 1] < extent[3])
    )
    if zlimit:
        filter = filter & (
                (zlimit[0] < coords[::, 2]) &
                (coords[::, 2] < zlimit[1])

        )
    print("before", coords.shape)
    if data is not None:
        data = data[filter]
    coords = coords[filter]

    print("after", coords.shape)
    if data is not None:
        return coords, data
    return coords


def create_2d_slice(center: List[float], extent, coords: np.ndarray, property_name: str, property_data: np.ndarray,
                    resolution: int,
                    method="nearest") -> np.ndarray:
    cut_axis = 2  # Z

    coords, property_data = filter_3d(coords, extent, property_data)
    if property_name == "Temperatures":
        print("calculating temperatures")
        property_data = np.array([calculate_T(u) for u in property_data])
    xrange = np.linspace(extent[0], extent[1], resolution)
    yrange = np.linspace(extent[2], extent[3], resolution)
    gx, gy, gz = np.meshgrid(xrange, yrange, center[cut_axis])
    print("interpolating")
    grid = griddata(coords, property_data, (gx, gy, gz), method=method)[::, ::, 0]
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
