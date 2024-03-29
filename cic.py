import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numba import njit

from readfiles import read_file

Extent = Tuple[float, float, float, float]


@njit(parallel=True)
def cic_deposit(X, Y, ngrid, periodic=True) -> np.ndarray:
    rho = np.zeros((ngrid, ngrid))
    for x, y in zip(X, Y):
        if periodic:
            x = np.fmod(1.0 + x, 1.0)
            y = np.fmod(1.0 + y, 1.0)
        else:
            if not (0 < x < 1) or not (0 < y < 1):
                continue
        il = int(np.floor(x * ngrid))
        ir = (il + 1) % ngrid
        jl = int(np.floor(y * ngrid))
        jr = (jl + 1) % ngrid
        dx = x * ngrid - float(il)
        dy = y * ngrid - float(jl)
        rho[il, jl] += (1 - dx) * (1 - dy)
        rho[il, jr] += (1 - dx) * dy
        rho[ir, jl] += dx * (1 - dy)
        rho[ir, jr] += dx * dy
    rhomean = len(X) / ngrid ** 2
    return rho / rhomean - 1


def cic_range(
    X: np.ndarray,
    Y: np.ndarray,
    ngrid: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *args,
    **kwargs
) -> Tuple[np.ndarray, Extent]:
    xrange = xmax - xmin
    yrange = ymax - ymin
    Xs = (X - xmin) / xrange
    Ys = (Y - ymin) / yrange
    print(Xs.min(), Xs.max())
    print(Ys.min(), Ys.max())
    print((60 - ymin) / yrange)
    extent = (xmin, xmax, ymin, ymax)
    rho = cic_deposit(Xs, Ys, ngrid, *args, **kwargs)
    return rho, extent


def cic_from_radius(
    X: np.ndarray,
    Y: np.ndarray,
    ngrid: int,
    x_center: float,
    y_center: float,
    radius: float,
    *args,
    **kwargs
) -> Tuple[np.ndarray, Extent]:
    return cic_range(
        X,
        Y,
        ngrid,
        x_center - radius,
        x_center + radius,
        y_center - radius,
        y_center + radius,
        *args,
        **kwargs
    )


def plot_cic(rho: np.ndarray, extent: Extent, title: str):
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    data = 1.1 + rho
    i = ax.imshow(data.T, norm=LogNorm(), extent=extent, origin="lower")
    ax.set_title(title)

    fig.colorbar(i)
    plt.savefig((Path("~/tmp").expanduser() / title).with_suffix(".fig.png"))

    plt.show()

    cmap = plt.cm.viridis
    data = np.log(data)
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    image = cmap(norm(data.T))
    plt.imsave(
        (Path("~/tmp").expanduser() / title).with_suffix(".png"), image, origin="lower"
    )
    # ax.hist2d(df.X, df.Y, bins=500, norm=LogNorm())
    # ax.hist2d(df2.X, df2.Y, bins=1000, norm=LogNorm())


if __name__ == "__main__":
    input_file = Path(sys.argv[1])
    df_ref, _ = read_file(input_file)
    rho, extent = cic_from_radius(
        df_ref.X.to_numpy(), df_ref.Y.to_numpy(), 1500, 48.8, 57, 1, periodic=False
    )
    # rho, extent = cic_range(df_ref.X.to_numpy(), df_ref.Y.to_numpy(), 800, 0, 85.47, 0, 85.47, periodic=False)

    plot_cic(
        rho, extent, title=str(input_file.relative_to(input_file.parent.parent).name)
    )
