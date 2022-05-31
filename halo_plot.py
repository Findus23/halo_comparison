from typing import List

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from pyvista import Axes


def main():
    rows = ["shannon", "DB8", "DB4", "DB2"]
    offset = 2
    columns = [128, 256, 512]
    fig: Figure = plt.figure(figsize=(9, 9))
    axes: List[List[Axes]] = fig.subplots(len(rows), len(columns), sharex=True, sharey=True)
    with h5py.File("vis.cache.hdf5") as vis_out:
        vmin, vmax = vis_out["vmin_vmax"]
        print(vmin, vmax)
        for i, waveform in enumerate(rows):
            for j, resolution in enumerate(columns):
                ax = axes[i][j]
                rho = np.asarray(vis_out[f"{waveform}_{resolution}_rho"])
                extent = list(vis_out[f"{waveform}_{resolution}_extent"])
                mass = vis_out[f"{waveform}_{resolution}_mass"]
                vmin_scaled = (vmin + offset) * mass
                vmax_scaled = (vmax + offset) * mass
                rho = (rho + offset) * mass

                img = ax.imshow(rho.T, norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled), extent=extent,origin="lower")
                print(img)
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
