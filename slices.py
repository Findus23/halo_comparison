from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from temperatures import calculate_T
from utils import create_figure


def create_2d_slice(
        input_file: Path, center: List[float], property: str, axis="Z", thickness=3, method="nearest"
):
    axis_names = ["X", "Y", "Z"]
    cut_axis = axis_names.index(axis)
    limits = {
        "X": (46, 52),
        "Y": (54, 60),
        "Z": (center[cut_axis] - 10, center[cut_axis] + 10)
    }
    with h5py.File(input_file) as f:
        pt0 = f["PartType0"]
        coords = pt0["Coordinates"][:]
        energies = pt0["InternalEnergies"][:]
        entropies = pt0["Entropies"][:]

        print((center[cut_axis] - thickness < coords[::, cut_axis]).shape)
        # in_slice = (center[cut_axis] - thickness < coords[::, cut_axis]) & (
        #         coords[::, cut_axis] < center[cut_axis] + thickness)
        # print("got slice")
        # coords_in_slice = coords[in_slice]
        # data_in_slice = data[in_slice]
        filter = (
                (limits["X"][0] < coords[::, 0]) &
                (coords[::, 0] < limits["X"][1]) &

                (limits["Y"][0] < coords[::, 1]) &
                (coords[::, 1] < limits["Y"][1]) &

                (limits["Z"][0] < coords[::, 2]) &
                (coords[::, 2] < limits["Z"][1])
        )
        print("before", coords.shape)
        energies = energies[filter]
        entropies = entropies[filter]
        coords = coords[filter]

        print("after", coords.shape)
        print("calculating temperatures")
        temperatures = np.array([calculate_T(u) for u in energies])

        other_axis = {"X": ("Y", "Z"), "Y": ("X", "Z"), "Z": ("X", "Y")}
        x_axis_label, y_axis_label = other_axis[axis]
        x_axis = axis_names.index(x_axis_label)
        y_axis = axis_names.index(y_axis_label)
        xrange = np.linspace(coords[::, x_axis].min(), coords[::, x_axis].max(), 1000)
        yrange = np.linspace(coords[::, y_axis].min(), coords[::, y_axis].max(), 1000)
        gx, gy, gz = np.meshgrid(xrange, yrange, center[cut_axis])
        print("interpolating")
        grid = griddata(coords, temperatures, (gx, gy, gz), method=method)[::, ::, 0]
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
        fig.savefig(Path("~/tmp/slice.png").expanduser(), dpi=300)
        plt.show()
        exit()
