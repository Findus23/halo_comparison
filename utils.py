from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def print_progress(i, total, extra_data=""):
    print(f"{i} of {total} ({extra_data})" + " " * 20, end="\r" if i != total else "\n", flush=True)


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
    with(dir / "swift.log").open() as f:
        last_line = f.readlines()[-1]
    print(last_line)
    assert "main: done. Bye." in last_line
    seconds = float(last_line[1:].split("]")[0])
    print(f"Runtime: {seconds / 60 / 60:.2f} hours")
