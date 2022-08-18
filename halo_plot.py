from pathlib import Path
from sys import argv
from typing import List

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from halo_vis import Coords
from paths import base_dir, vis_datafile, has_1024_simulations
from read_vr_files import read_velo_halos
from utils import figsize_from_page_fraction, rowcolumn_labels, waveforms


def coord_to_2d_extent(coords: Coords):
    radius, X, Y, Z = coords
    return X - radius, X + radius, Y - radius, Y + radius


def in_area(coords: Coords, xobj, yobj, zobj, factor=1.3) -> bool:
    radius, xcenter, ycenter, zcenter = coords
    radius *= factor
    return (
            (xcenter - radius < xobj < xcenter + radius)
            and (ycenter - radius < yobj < ycenter + radius)
            and (zcenter - radius < zobj < zcenter + radius)
    )


def main():
    offset = 2
    resolutions = [128, 256, 512]
    if argv[1] == "box":
        initial_halo_id = 0
    else:
        initial_halo_id = int(argv[1])
    is_box = not initial_halo_id
    if has_1024_simulations:
        resolutions.append(1024)
    fig: Figure = plt.figure(
        figsize=figsize_from_page_fraction(columns=2, height_to_width=1 if is_box else 1.3)
    )
    axes: List[List[Axes]] = fig.subplots(
        len(waveforms) if is_box else len(waveforms) * 2, 4, sharex="row", sharey="row"
    )
    with h5py.File(vis_datafile) as vis_out:
        halo_group = vis_out[str(initial_halo_id)]

        vmin, vmax = halo_group["vmin_vmax"]
        print(vmin, vmax)
        for i, waveform in enumerate(waveforms):
            for j, resolution in enumerate(resolutions):
                dir = base_dir / f"{waveform}_{resolution}_100"
                halos = read_velo_halos(dir)
                if is_box:
                    ax_both = axes[i][j]
                else:
                    ax_both = axes[i * 2][j]
                    ax_without = axes[i * 2 + 1][j]
                dataset_group = halo_group[f"{waveform}_{resolution}"]
                rho = np.asarray(dataset_group["rho"])
                # radius, X, Y, Z
                coords: Coords = tuple(dataset_group["coords"])
                radius, X, Y, Z = coords
                mass = dataset_group["mass"][()]  # get scalar value from Dataset
                main_halo_id = dataset_group["halo_id"][()] if initial_halo_id else None
                vmin_scaled = (vmin + offset) * mass
                vmax_scaled = (vmax + offset) * mass
                rho = (rho + offset) * mass
                extent = coord_to_2d_extent(coords)
                for ax in [ax_both, ax_without] if not is_box else [ax_both]:
                    img = ax.imshow(
                        rho.T.T,
                        norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled),
                        extent=extent,
                        origin="lower",
                        cmap="Greys",
                    )  # ax.set_axis_off()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if not is_box:
                        ax.set_ylim(Y - radius * .6, Y + radius * .6)
                    if j == 0:
                        scalebar = AnchoredSizeBar(
                            ax.transData,
                            1,
                            "1 Mpc",
                            "lower left",
                            # pad=0.1,
                            color="black",
                            frameon=False,
                            # size_vertical=1
                        )
                        ax.add_artist(scalebar)

                if not is_box:
                    found_main_halo = False
                    for halo_id, halo in halos.iterrows():
                        if halo["Vmax"] > 75:
                            if in_area(coords, halo.X, halo.Y, halo.Z):
                                if halo_id == main_halo_id:
                                    color = "C2"
                                elif halo["Structuretype"] > 10:
                                    color = "C0"
                                else:
                                    color = "C1"
                                if halo_id == main_halo_id:
                                    found_main_halo = True
                                    print("plotting main halo")
                                circle = Circle(
                                    (halo.Y - Y + X, halo.X - X + Y),
                                    halo["Rvir"],
                                    zorder=10,
                                    linewidth=1,
                                    edgecolor=color,
                                    fill=None,
                                    alpha=0.2,
                                )
                                ax_both.add_artist(circle)

                    assert found_main_halo
                print(img)

            #     break
            # break
    ylabels = [item for item in waveforms for _ in range(2)]
    rowcolumn_labels(axes, ylabels, isrow=True)
    rowcolumn_labels(axes, resolutions, isrow=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(Path(f"~/tmp/halo_plot_{initial_halo_id}.pdf").expanduser())
    fig.savefig(Path(f"~/tmp/halo_plot_{initial_halo_id}.png").expanduser(), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
