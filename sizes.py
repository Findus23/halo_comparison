from sys import argv

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file = argv[1]

df = pd.read_csv(file)

print(df)

# df = df.iloc

fig: Figure = plt.figure()
ax: Axes = fig.gca()
# ax.scatter(df["ref_sizes"], df["comp_sizes"], s=1, alpha=.3)
ax.scatter(df["ref_masses"], df["comp_masses"], s=1, alpha=.3)

ax.set_xscale("log")
ax.set_xlabel("reference size")
ax.set_ylabel("comparison size")
ax.set_yscale("log")

fig2: Figure = plt.figure()
ax2: Axes = fig2.gca()

ax2.hist(df["distances"][df["distances"] < 50], bins=100)
ax2.set_xlabel("distance/R_vir_ref")
for a in [ax, ax2]:
    a.set_title(file)

plt.show()
