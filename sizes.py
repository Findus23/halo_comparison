from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

file = Path(argv[1])

df = pd.read_csv(file)

with pd.option_context('display.max_rows', None):
    print(df[["ref_npart", "comp_npart", "ref_cNFW", "comp_cNFW"]])
# df = df.iloc

fig: Figure = plt.figure()
ax: Axes = fig.gca()
# hist2d, log?

x_col = "ref_cNFW"
y_col = "comp_cNFW"

# x_col = "ref_Mvir"
# y_col = "comp_Mvir"

min_x = min([min(df[x_col]), min(df[y_col])])
max_x = max([max(df[x_col]), max(df[y_col])])

bins = np.geomspace(min_x, max_x, 100)

# ax.scatter(df["ref_sizes"], df["comp_sizes"], s=1, alpha=.3)
# ax.scatter(df[x_col], df[y_col], s=1, alpha=.3)
_, _, _, hist = ax.hist2d(df[x_col], df[y_col], bins=(bins, bins), norm=LogNorm())

# ax.set_xscale("log")
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
# ax.set_yscale("log")

fig.colorbar(hist)

ax.loglog([min_x, max_x], [min_x, max_x], linewidth=1, color="C2")

fig2: Figure = plt.figure()
ax2: Axes = fig2.gca()

ax2.hist(df["distance"][df["distance"] < 50], bins=100)
ax2.set_xlabel("distance/R_vir_ref")
for a in [ax, ax2]:
    a.set_title(file.name)

fig.savefig(Path(f"~/tmp/comparison_{file.stem}.pdf").expanduser())
fig2.savefig(Path(f"~/tmp/distances_{file.stem}.pdf").expanduser())
fig.suptitle
plt.show()
