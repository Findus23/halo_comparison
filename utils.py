from pathlib import Path
from string import ascii_uppercase
from typing import Tuple

import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

waveforms = ["DB2", "DB4", "DB8", "shannon"]


def print_progress(i, total, extra_data=""):
    print(
        f"{i} of {total} ({extra_data})" + " " * 20,
        end="\r" if i != total else "\n",
        flush=True,
    )


def memory_usage(df: pd.DataFrame):
    bytes_used = df.memory_usage(index=True).sum()
    return bytes_used / 1024 / 1024


def create_figure() -> Tuple[Figure, Axes]:
    """
    helper function for matplotlib OOP interface with proper typing
    """
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()
    return fig, ax


def read_swift_config(dir: Path):
    with (dir / "used_parameters.yml").open() as f:
        return yaml.safe_load(f)


def print_wall_time(dir: Path):
    with (dir / "swift.log").open() as f:
        last_line = f.readlines()[-1]
    print(last_line)
    assert "main: done. Bye." in last_line
    seconds = float(last_line[1:].split("]")[0])
    print(f"Runtime: {seconds / 60 / 60:.2f} hours")


def figsize_from_page_fraction(columns=1, height_to_width=3 / 4):
    cm = 1 / 2.54  # centimeters in inches
    # \printinunitsof{cm}\prntlen{\linewidth}
    two_column_width = 17.85162  # cm
    one_colum_width = 8.5744  # cm
    upscale_factor = 1.3

    width = two_column_width if columns == 2 else one_colum_width
    height = width * height_to_width
    return width * cm * upscale_factor, height * cm * upscale_factor


def rowcolumn_labels(axes, labels, isrow: bool, pad=5) -> None:
    """
    based on https://stackoverflow.com/a/25814386/4398037
    """
    axs = axes[:, 0] if isrow else axes[0]
    for ax, label in zip(axs, labels):
        ax: Axes
        if isrow:
            xy = (0, 0.5)
            xytext = (-ax.yaxis.labelpad - pad, 0)
            xycoords = ax.yaxis.label
            ha = "right"
            va = "center"
        else:
            xy = (0.5, 1)
            xytext = (0, pad)
            xycoords = "axes fraction"
            ha = "center"
            va = "baseline"
        ax.annotate(
            label,
            xy=xy,
            xytext=xytext,
            xycoords=xycoords,
            textcoords="offset points",
            size="medium",
            ha=ha,
            va=va,
            rotation=90
        )


def tex_fmt(format_str: str, *args) -> str:
    for i, arg in enumerate(args):
        format_str = format_str.replace(ascii_uppercase[i] * 2, str(arg))
    return format_str
