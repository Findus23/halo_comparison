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

        fig.suptitle(f"Power Spectra {time}")
        fig.tight_layout()

    elif mode == "cross":
        combination_list = list(itertools.combinations(resolutions, 2))
        subfigs = fig.subplots(
            len(combination_list), 1, sharex=True, sharey=True
        ).flatten()
        for j, (res1, res2) in enumerate(combination_list):
            ax: Axes = subfigs[j]
            ax.set_xlabel("k [Mpc$^{-1}$]")
            ax.set_ylabel("C")
            ax.text(
                0.02,
                0.93,
                f"{res1} vs {res2}",
                size=10,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            for i, waveform in enumerate(waveforms):
                data = spectra_data(waveform, res1, res2, Lbox, time)
                k = data["k [Mpc]"]
                pcross = data["Pcross"]

                ax.semilogx(k, pcross, color=colors[i], label=waveform)
            ax.set_xlim(right=k0 * res1)
            ax.set_ylim(0.8, 1.02)
            ax.legend()

        fig.suptitle(f"Cross Spectra {time}")
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
