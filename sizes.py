import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file = "DB2_128_100_DB2_512_100.csv"

df = pd.read_csv(file)

print(df)

# df = df.iloc

fig: Figure = plt.figure()
ax: Axes = fig.gca()
ax.scatter(df["ref_sizes"], df["comp_sizes"], s=1, alpha=.3)
# ax.scatter(df["ref_masses"], df["comp_masses"], s=3)

ax.set_title(file)
ax.set_xscale("log")
ax.set_xlabel("reference size")
ax.set_ylabel("comparison size")
ax.set_yscale("log")
plt.show()
