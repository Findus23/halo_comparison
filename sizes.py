import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

df = pd.read_csv("sizes.csv")

print(df)

# df = df.iloc

fig: Figure = plt.figure()
ax: Axes = fig.gca()
ax.scatter(df["ref_sizes"], df["comp_sizes"], s=1, alpha=.3)
# ax.scatter(df["ref_masses"], df["comp_masses"], s=3)

ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
