from pathlib import Path
from sys import argv

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file = Path(argv[1])

df = pd.read_csv(file)

with pd.option_context('display.max_rows', None):
    print(df[["ref_npart","comp_npart","ref_cNFW","comp_cNFW"]])
# df = df.iloc

fig: Figure = plt.figure()
ax: Axes = fig.gca()

x_col = "ref_cNFW"
y_col = "comp_cNFW"

# ax.scatter(df["ref_sizes"], df["comp_sizes"], s=1, alpha=.3)
ax.scatter(df[x_col], df[y_col], s=1, alpha=.3)

# ax.set_xscale("log")
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
# ax.set_yscale("log")

min_x = min([min(df[x_col]), min(df[y_col])])
max_x = max([max(df[x_col]), max(df[y_col])])

ax.plot([min_x, max_x], [min_x, max_x], linewidth=1, color="C2")

fig2: Figure = plt.figure()
ax2: Axes = fig2.gca()

ax2.hist(df["distance"][df["distance"] < 50], bins=100)
ax2.set_xlabel("distance/R_vir_ref")
for a in [ax, ax2]:
    a.set_title(file.name)

plt.show()
