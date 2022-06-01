from typing import List

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from pyvista import Axes

from cic import Extent
from paths import base_dir, vis_datafile
from read_vr_files import read_velo_halos


def increase_extent_1d(xmin: float, xmax: float, factor: float):
    xrange = xmax - xmin
    xcenter = (xmax + xmin) / 2
    return xcenter - xrange / 2 * factor, xcenter + xrange / 2 * factor


def increase_extent(extent: Extent, factor: float) -> Extent:
    xmin, xmax, ymin, ymax = extent
    xmin, xmax = increase_extent_1d(xmin, xmax, factor)
    ymin, ymax = increase_extent_1d(ymin, ymax, factor)
    return xmin, xmax, ymin, ymax


def in_extent(extent: Extent, X, Y, factor=2) -> bool:
    xmin, xmax, ymin, ymax = increase_extent(extent, factor)
    return (xmin < X < xmax) and (ymin < Y < ymax)


def main():
    rows = ["shannon", "DB8", "DB4", "DB2"]
    offset = 2
    columns = [128, 256, 512]
    fig: Figure = plt.figure(figsize=(9, 9))
    axes: List[List[Axes]] = fig.subplots(len(rows), len(columns), sharex=True, sharey=True)
    with h5py.File(vis_datafile) as vis_out:
        vmin, vmax = vis_out["vmin_vmax"]
        print(vmin, vmax)
        for i, waveform in enumerate(rows):
            for j, resolution in enumerate(columns):
                dir = base_dir / f"{waveform}_{resolution}_100"
                halos = read_velo_halos(dir)
                ax = axes[i][j]
                rho = np.asarray(vis_out[f"{waveform}_{resolution}_rho"])
                extent = tuple(vis_out[f"{waveform}_{resolution}_extent"])
                mass = vis_out[f"{waveform}_{resolution}_mass"][()]  # get scalar value from Dataset
                main_halo_id = vis_out[f"{waveform}_{resolution}_halo_id"][()]
                vmin_scaled = (vmin + offset) * mass
                vmax_scaled = (vmax + offset) * mass
                rho = (rho + offset) * mass

                img = ax.imshow(rho.T, norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled), extent=extent,
                                origin="lower")
                for halo_id, halo in halos.iterrows():
                    if halo["Vmax"] > 135:
                        if in_extent(extent, halo.X, halo.Y):
                            color = "red" if halo_id == main_halo_id else "white"
                            if halo_id == main_halo_id:
                                print(halo_id == main_halo_id, halo_id, main_halo_id, halo["Rvir"])
                                print("plotting main halo")
                            circle = Circle(
                                (halo.X, halo.Y),
                                halo["Rvir"], zorder=10,
                                linewidth=1, edgecolor=color, fill=None, alpha=.2
                            )
                            ax.add_artist(circle)

                print(img)
            #     break
            # break
    pad = 5
    # based on https://stackoverflow.com/a/25814386/4398037
    for ax, col in zip(axes[0], columns):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    fig.tight_layout()
    fig.subplots_adjust(right=0.825)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    fig.savefig("halo_plot.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
