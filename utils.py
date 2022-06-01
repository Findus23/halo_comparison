from typing import Tuple

import pandas as pd
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

