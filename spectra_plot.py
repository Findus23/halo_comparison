import itertools
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paths import base_dir

Lbox = 100
k0 = 2 * 3.14159265358979323846264338327950 / Lbox
resolutions = [128, 256, 512]
waveforms = ["DB2", "DB4", "DB8", "shannon"]

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
colors = ["C1", "C2", "C3", "C4"]


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
        raise ValueError(f"invalid time ({time}) should be (ics|end)")

    # only consider rows above resolution limit
    spectra_data = spectra_data[spectra_data["k [Mpc]"] >= k0]

    return spectra_data


def create_plot(mode, time, show=True):
    fig: Figure = plt.figure(figsize=(9, 9))
    if mode == "power":
        subfigs = fig.subplots(len(waveforms), 1, sharex=True, sharey=True).flatten()
        for i, waveform in enumerate(waveforms):
            ax: Axes = subfigs[i]
            ax.set_xlabel("k [Mpc$^{-1}$]")
            ax.set_ylabel("P")
            ax.text(
                0.02,
                0.93,
                waveform,
                size=10,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            for j, resolution in enumerate(resolutions):
                data = spectra_data(waveform, resolution, resolution, Lbox, time)
                k = data["k [Mpc]"]
                p1 = data["P1"]
                p1_error = data["err. P1"]

                ax.loglog(k, p1, color=colors[j])
                ax.axvline(
                    k0 * resolution,
                    color=colors[j],
                    linestyle="dashed",
                    label=f"{resolution}",
                )
            ax.legend()

        # fig.suptitle(f"Power Spectra {time}") #Not needed for paper
        fig.tight_layout()

    elif mode == "cross":
        combination_list = list(itertools.combinations(resolutions, 2))
        subfigs = fig.subplots(
            len(waveforms), 2, sharex=True, sharey=True
        ).flatten()
        fig.subplots_adjust(wspace = 0, hspace = 0)
        for i, waveform in enumerate(waveforms):
            ax_ics: Axes = subfigs[2 * i]
            ax_end: Axes = subfigs[2 * i + 1]

            if i == len(waveforms) - 1:
                ax_ics.set_xlabel("k [Mpc$^{-1}$]")
            ax_ics.set_ylabel("C")
            ax_ics.text(
                0.02,
                0.85,
                f"{waveform}",
                size=13,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_ics.transAxes,
            )
            ax_ics.set_xticklabels([])
            ax_ics.set_yticklabels([])
            if i == len(waveforms) - 1:
                ax_end.set_xlabel("k [Mpc$^{-1}$]")
            # ax_end.set_ylabel("C")
            ax_end.text(
                0.02,
                0.85,
                f"{waveform}",
                size=13,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_end.transAxes,
            )
            ax_end.set_xticklabels([])
            ax_end.set_yticklabels([])
            for j, (res1, res2) in enumerate(combination_list):
                ics_data = spectra_data(waveform, res1, res2, Lbox, 'ics')
                ics_k = ics_data["k [Mpc]"]
                ics_pcross = ics_data["Pcross"]

                ax_ics.semilogx(ics_k, ics_pcross, color=colors[j], label=f'{res1} vs {res2}')

                end_data = spectra_data(waveform, res1, res2, Lbox, 'end')
                end_k = end_data["k [Mpc]"]
                end_pcross = end_data["Pcross"]

                ax_end.semilogx(end_k, end_pcross, color=colors[j], label=f'{res1} vs {res2}')

            ax_end.set_xlim(right=k0 * res1)
            ax_end.set_ylim(0.8, 1.02)
            ax_end.legend()

        # fig.suptitle(f"Cross Spectra {time}") #Not needed for paper
        fig.tight_layout()
    fig.savefig(Path(f"~/tmp/spectra_{time}_{mode}.pdf").expanduser())
    if show:
        plt.show()


def main():
    if len(argv) < 2:
        print("run spectra_plot.py [ics|end] [power|cross] or spectra_plot.py all")
        exit(1)
    if argv[1] == "all":
        for time in ["ics", "end"]:
            for mode in ["power", "cross"]:
                create_plot(mode, time, show=False)
        return
    time = argv[1]
    mode = argv[2]
    create_plot(mode, time)


if __name__ == "__main__":
    main()
