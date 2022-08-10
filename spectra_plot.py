import itertools
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import XTick, YTick
from matplotlib.figure import Figure

from paths import base_dir
from utils import figsize_from_page_fraction, waveforms

Lbox = 100
h = 0.690021
k0 = 3.14159265358979323846264338327950 / Lbox
resolutions = [128, 256, 512, 1024]

# Careful: k is actually in Mpc^-1, the column is just named weirdly.
columns = [
    "k [Mpc]",
    "Pcross",
    "P1",
    "err. P1",
    "P2",
    "err. P2",
    "P2-1",
    "err. P2-1",
    "modes in bin",
]

# linestyles = ["solid", "dashed", "dotted"]
colors = [f"C{i}" for i in range(10)]


# colors = ["C1", "C2", "C3", "C4"]


def spectra_data(
        waveform: str, resolution_1: int, resolution_2: int, Lbox: int, time: str
):
    dir = base_dir / f"spectra/{waveform}_{Lbox}"

    if time == "ics":
        spectra_data = pd.read_csv(
            f"{dir}/{waveform}_{Lbox}_ics_{resolution_1}_{resolution_2}_cross_spectrum.txt",
            sep=" ",
            skipinitialspace=True,
            header=None,
            names=columns,
            skiprows=1,
        )
    elif time == "z=1":
        spectra_data = pd.read_csv(
            f"{dir}/{waveform}_{Lbox}_a2_{resolution_1}_{resolution_2}_cross_spectrum.txt",
            sep=" ",
            skipinitialspace=True,
            header=None,
            names=columns,
            skiprows=1,
        )
    elif time == "end":
        spectra_data = pd.read_csv(
            f"{dir}/{waveform}_{Lbox}_a4_{resolution_1}_{resolution_2}_cross_spectrum.txt",
            sep=" ",
            skipinitialspace=True,
            header=None,
            names=columns,
            skiprows=1,
        )
    else:
        raise ValueError(f"invalid time ({time}) should be (ics|z=1|end)")

    # only consider rows above resolution limit
    spectra_data = spectra_data[spectra_data["k [Mpc]"] >= k0]

    return spectra_data


def create_plot(mode):
    fig: Figure
    combination_list = list(itertools.combinations(resolutions, 2))
    fig, axes = plt.subplots(
        len(waveforms), 3, sharex=True, sharey=True,
        figsize=figsize_from_page_fraction(columns=2),
    )
    crossings = np.zeros((len(waveforms), len(combination_list)))
    for i, waveform in enumerate(waveforms):
        ax_ics: Axes = axes[i][0]
        ax_z1: Axes = axes[i][1]
        ax_end: Axes = axes[i][2]
        axes_names = {
            # TODO: better names
            ax_ics: "ics",
            ax_z1: "z=1",
            ax_end: "z=0"
        }
        bottom_row = i == len(waveforms) - 1
        top_row = i == 0
        # for is_end, ax in enumerate([ax_ics, ax_z1]):
        for is_end, ax in enumerate([ax_ics, ax_z1, ax_end]):
            if bottom_row:
                ax.set_xlabel("k [Mpc$^{-1}$]")
            ax.text(
                0.01,
                0.85,
                f"{waveform}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            ax.text(
                0.99,
                0.85,
                axes_names[ax],
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            for j, res in enumerate(resolutions[:-1] if mode == "cross" else resolutions):
                ax.axvline(
                    k0 * res,
                    color=colors[j],
                    linestyle="dashed",
                    label=f"$k_\\mathrm{{ny, {res}}}$" if mode == "power" else None,
                )
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])

        if mode == "power":
            ax_ics.set_ylabel("$\\mathrm{{P}}_\\mathrm{{X}}$ / $\\mathrm{{P}}_{{1024}}$")
            for j, resolution in enumerate(resolutions):
                ics_data = spectra_data(waveform, resolution, resolution, Lbox, "ics")
                ics_k = ics_data["k [Mpc]"]
                ics_p1 = ics_data["P1"]
                comp_data = spectra_data(waveform, resolutions[-1], resolutions[-1], Lbox, "ics")
                comp_p1 = comp_data["P1"]
                ics_p1 /= comp_p1

                end_data = spectra_data(waveform, resolution, resolution, Lbox, "end")
                end_k = end_data["k [Mpc]"]
                end_p1 = end_data["P1"]
                comp_data = spectra_data(waveform, resolutions[-1], resolutions[-1], Lbox, "end")
                comp_p1 = comp_data["P1"]
                end_p1 /= comp_p1

                z1_data = spectra_data(waveform, resolution, resolution, Lbox, "z=1")
                z1_k = z1_data["k [Mpc]"]
                z1_p1 = z1_data["P1"]
                comp_data = spectra_data(waveform, resolutions[-1], resolutions[-1], Lbox, 'z=1')
                comp_p1 = comp_data["P1"]
                z1_p1 /= comp_p1

                ax_ics.semilogx(ics_k, ics_p1, color=colors[j])
                ax_z1.semilogx(z1_k, z1_p1, color=colors[j])
                ax_end.semilogx(end_k, end_p1, color=colors[j])
                for ax in [ax_ics, ax_z1, ax_end]:
                    ax.set_ylim(0.9, 1.10)
                    ax.set_axisbelow(True)
                    ax.grid(color='black', linestyle=':', linewidth=0.5, alpha=0.5)


        # fig.suptitle(f"Power Spectra {time}") #Not needed for paper
        # fig.tight_layout()

        elif mode == "cross":
            ax_ics.set_ylabel("C")
            # ax_end.set_ylabel("C")
            for j, (res1, res2) in enumerate(combination_list):
                ics_data = spectra_data(waveform, res1, res2, Lbox, 'ics')
                ics_k = ics_data["k [Mpc]"]
                ics_pcross = ics_data["Pcross"]

                ax_ics.semilogx(ics_k, ics_pcross, color=colors[j + 3], label=f'{res1} vs {res2}')

                z1_data = spectra_data(waveform, res1, res2, Lbox, 'z=1')
                z1_k = z1_data["k [Mpc]"]
                z1_pcross = z1_data["Pcross"]

                ax_z1.semilogx(z1_k, z1_pcross, color=colors[j + 3], label=f'{res1} vs {res2}')

                end_data = spectra_data(waveform, res1, res2, Lbox, 'end')
                end_k = end_data["k [Mpc]"]
                end_pcross = end_data["Pcross"]

                ax_end.semilogx(end_k, end_pcross, color=colors[j + 3], label=f'{res1} vs {res2}')

                # #Put this here to enable changing time of crossing measurement more easily
                smaller_res = min(res1, res2)
                crossing_index = np.searchsorted(end_k.to_list(), k0 * smaller_res)  # change here
                crossing_value = end_pcross[crossing_index]  # and here
                crossings[i][j] = crossing_value

                for ax in [ax_ics, ax_z1, ax_end]:
                    ax.set_axisbelow(True)
                    ax.grid(color='black', linestyle=':', linewidth=0.5, alpha=0.5)

            ax_end.set_xlim(right=k0 * resolutions[-1])
            ax_end.set_ylim(0.8, 1.02)
        if bottom_row:
            # ax_z1.legend()
            ax_ics.legend(loc='lower left')
        if not bottom_row:
            last_xtick: XTick = ax_ics.yaxis.get_major_ticks()[0]
            last_xtick.set_visible(False)

        # fig.suptitle(f"Cross Spectra {time}") #Not needed for paper
        # fig.tight_layout()
    print(crossings)
    crossings_df = pd.DataFrame(crossings, columns=combination_list, index=waveforms)
    # print(crossings_df.to_markdown())
    print(crossings_df.to_latex())
    fig.tight_layout()
    fig.subplots_adjust(wspace=0,hspace=0)

    fig.savefig(Path(f"~/tmp/spectra_{mode}.pdf").expanduser())


def main():
    if len(argv) < 2:
        print("run spectra_plot.py [power|cross] or spectra_plot.py all")
        exit(1)
    if argv[1] == "all":
        for mode in ["power", "cross"]:
            create_plot(mode)
        plt.show()
        return
    mode = argv[1]
    create_plot(mode)
    plt.show()


if __name__ == "__main__":
    main()
