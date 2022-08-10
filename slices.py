from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from utils import create_figure


def create_2d_slice(
    input_file: Path, center: List[float], property: str, axis="Z", thickness=3
):
    axis_names = ["X", "Y", "Z"]
    cut_axis = axis_names.index(axis)
    with h5py.File(input_file) as f:
        pt0 = f["PartType0"]
        coords = pt0["Coordinates"]
        data = pt0[property]
        print((center[cut_axis] - thickness < coords[::, cut_axis]).shape)
        # in_slice = (center[cut_axis] - thickness < coords[::, cut_axis]) & (
        #         coords[::, cut_axis] < center[cut_axis] + thickness)
        # print("got slice")
        # coords_in_slice = coords[in_slice]
        # data_in_slice = data[in_slice]
        print("stats")
        other_axis = {"X": ("Y", "Z"), "Y": ("X", "Z"), "Z": ("X", "Y")}
        x_axis_label, y_axis_label = other_axis[axis]
        x_axis = axis_names.index(x_axis_label)
        y_axis = axis_names.index(y_axis_label)
        xrange = np.linspace(coords[::, x_axis].min(), coords[::, x_axis].max(), 1000)
        yrange = np.linspace(coords[::, y_axis].min(), coords[::, y_axis].max(), 1000)
        gx, gy, gz = np.meshgrid(xrange, yrange, center[cut_axis])
        print("interpolating")
        grid = griddata(coords, data, (gx, gy, gz), method="linear")[::, ::, 0]
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
            grid.T,
            norm=LogNorm(),
            interpolation="nearest",
            extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]],
        )
        ax.set_title(input_file.parent.stem)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        fig.colorbar(img, label=property)
        fig.tight_layout()
        plt.show()
        exit()
