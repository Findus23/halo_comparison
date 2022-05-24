from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numba import njit

from paths import base_dir
from readfiles import read_file


@njit()
def cic_deposit(X, Y, ngrid):
    rho = np.zeros((ngrid, ngrid))
    for x, y in zip(X, Y):
        x = np.fmod(1.0 + x, 1.0)
        y = np.fmod(1.0 + y, 1.0)
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


if __name__ == '__main__':
    reference_dir = Path(base_dir / f"shannon_512_100")
    df_ref, _ = read_file(reference_dir)


    fig: Figure = plt.figure()
    ax: Axes = fig.gca()

    print("start cic")
    rho = cic_deposit(df_ref.X.to_numpy() / 100, df_ref.Y.to_numpy() / 100, 2000)
    print("finished cic")
    data = 1.001 + rho
    i = ax.imshow(data, norm=LogNorm())
    fig.colorbar(i)
    plt.show()

    cmap = plt.cm.viridis
    data = np.log(data)
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    image = cmap(norm(data))
    plt.imsave("out.png", image)
    # ax.hist2d(df.X, df.Y, bins=500, norm=LogNorm())
    # ax.hist2d(df2.X, df2.Y, bins=1000, norm=LogNorm())
