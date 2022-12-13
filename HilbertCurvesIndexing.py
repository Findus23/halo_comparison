"""
originally created by Oliver Hahn
in HilbertCurvesIndexing.ipynb
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, Generator, PCG64

# dictionary containing the first order hilbert curves
from utils import figsize_from_page_fraction

# from matplotlib import rc
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Times New Roman']})
# rc('text', usetex=True)
# rc('axes', titlesize=24)
# rc('axes', labelsize=20)
# rc('axes', axisbelow=False)
# rc('lines',linewidth=2)
# # lines.markersize : 10
# rc('xtick', labelsize=16)
# rc('xtick.major',size=10)
# rc('xtick.minor',size=5)
# rc('xtick',direction='in')
# rc('ytick', labelsize=16)
# rc('ytick.major',size=10)
# rc('ytick.minor',size=5)
# rc('ytick',direction='in')
# rc('legend',fontsize='x-large')

base_shape = {
    "u": [np.array([0, 1]), np.array([1, 0]), np.array([0, -1])],
    "d": [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])],
    "r": [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])],
    "l": [np.array([-1, 0]), np.array([0, -1]), np.array([1, 0])],
}


def hilbert_curve(order, orientation):
    """
    Recursively creates the structure for a hilbert curve of given order
    """
    if order > 1:
        if orientation == "u":
            return (
                hilbert_curve(order - 1, "r")
                + [np.array([0, 1])]
                + hilbert_curve(order - 1, "u")
                + [np.array([1, 0])]
                + hilbert_curve(order - 1, "u")
                + [np.array([0, -1])]
                + hilbert_curve(order - 1, "l")
            )
        elif orientation == "d":
            return (
                hilbert_curve(order - 1, "l")
                + [np.array([0, -1])]
                + hilbert_curve(order - 1, "d")
                + [np.array([-1, 0])]
                + hilbert_curve(order - 1, "d")
                + [np.array([0, 1])]
                + hilbert_curve(order - 1, "r")
            )
        elif orientation == "r":
            return (
                hilbert_curve(order - 1, "u")
                + [np.array([1, 0])]
                + hilbert_curve(order - 1, "r")
                + [np.array([0, 1])]
                + hilbert_curve(order - 1, "r")
                + [np.array([-1, 0])]
                + hilbert_curve(order - 1, "d")
            )
        else:
            return (
                hilbert_curve(order - 1, "d")
                + [np.array([-1, 0])]
                + hilbert_curve(order - 1, "l")
                + [np.array([0, -1])]
                + hilbert_curve(order - 1, "l")
                + [np.array([1, 0])]
                + hilbert_curve(order - 1, "u")
            )
    else:
        return base_shape[orientation]


# test the functions
# if __name__ == '__main__':
#     order = 8
#     curve = hilbert_curve(order, 'u')
#     curve = np.array(curve) * 4
#     cumulative_curve = np.array([np.sum(curve[:i], 0) for i in range(len(curve)+1)])
#     # plot curve using plt
#     plt.plot(cumulative_curve[:, 0], cumulative_curve[:, 1])
# draw curve using turtle graphics
#     tt.setup(1920, 1000)
#     tt.pu()
#     tt.goto(-950, -490)
#     tt.pd()
#     tt.speed(0)
#     for item in curve:
#         tt.goto(tt.pos()[0] + item[0], tt.pos()[1] + item[1])
#     tt.done()


order = 6
curve = hilbert_curve(order, "u")
curve = np.array(curve) * 4
cumulative_curve_int = np.array([np.sum(curve[:i], 0) for i in range(len(curve) + 1)])
cumulative_curve = (
    np.array([np.sum(curve[:i], 0) for i in range(len(curve) + 1)]) + 2
) / 2 ** (order + 2)
# plot curve using plt
N = 2 ** (2 * order)
sublevel = order - 4

cmap = cm.get_cmap("jet")

fig = plt.figure(figsize=figsize_from_page_fraction(height_to_width=1))
t = {}
sublevel = 7
for i in range(2 ** (2 * sublevel)):
    il = i * N // (2 ** (2 * sublevel))
    ir = (i + 1) * N // 2 ** (2 * sublevel)
    plt.plot(
        cumulative_curve[il : ir + 1, 0],
        cumulative_curve[il : ir + 1, 1],
        lw=0.5,
        c=cmap(i / 2 ** (2 * sublevel)),
    )

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout()
plt.savefig(Path(f"~/tmp/hilbert_indexcolor.eps").expanduser())


order = 6
curve = hilbert_curve(order, "u")
curve = np.array(curve) * 4
cumulative_curve_int = np.array([np.sum(curve[:i], 0) for i in range(len(curve) + 1)])
cumulative_curve = (
    np.array([np.sum(curve[:i], 0) for i in range(len(curve) + 1)]) + 2
) / 2 ** (order + 2)
# plot curve using plt
N = 2 ** (2 * order)
sublevel = order - 4

cmap = cm.get_cmap("jet")


s = 10  # arbitrary lenght of each sequence drawn for every point

fig = plt.figure(figsize=figsize_from_page_fraction(height_to_width=1))
t = {}
sublevel = 7
for i in range(2 ** (2 * sublevel)):
    pcg = PCG64(1234)
    pcg = pcg.advance(i * s)
    rng = Generator(pcg)
    il = i * N // (2 ** (2 * sublevel))
    ir = (i + 1) * N // 2 ** (2 * sublevel)
    random_value = rng.random()
    plt.plot(
        cumulative_curve[il : ir + 1, 0],
        cumulative_curve[il : ir + 1, 1],
        lw=0.5,
        c=cmap(random_value),
    )

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout()
plt.savefig(Path(f"~/tmp/hilbert_indexcolor_scrambled.eps").expanduser())
plt.show()
