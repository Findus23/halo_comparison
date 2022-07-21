import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from halo_vis import get_comp_id
from paths import base_dir

num_bins = 5
bins = np.geomspace(450, 80000, num_bins + 1)
waveform = "shannon"
comparisons_dir = base_dir / "comparisons"


def read(mode, ref_res, comp_res):
    df = pd.read_csv(comparisons_dir / get_comp_id(waveform, ref_res, waveform, comp_res))
    # df = pd.read_csv(f"{mode}_{ref_res}_100_{mode}_{comp_res}_100.csv")
    print(min(df.ref_Mvir), max(df.ref_Mvir))

    digits = np.digitize(df.ref_Mvir, bins)
    bin_means = []
    for i in range(num_bins):
        values = np.where(digits == i + 1)
        in_bin = df.iloc[values]
        matches = np.array(in_bin.match)  # TODO: or instead fraction of halos that are matching?
        bin_means.append(matches.mean())
    return bin_means


rows = [[1] * num_bins]
resolutions = [128, 256, 512, 1024]
ref_res = 128
for res in resolutions:
    if res == ref_res:
        continue
    means = read(waveform, 128, res)
    rows.append(means)

data = np.array(rows).T

fig: Figure = plt.figure()
ax: Axes = fig.gca()
ax.set_xticks(range(len(resolutions)))
ax.set_xticklabels(resolutions)
ax.set_yticks(np.arange(len(bins)) - 0.5)
ax.set_yticklabels(["{:.2f}".format(a) for a in bins])

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        text = ax.text(y, x, "{:.2f}".format(data[x, y]), ha="center", va="center", color="w")

# print(data)
p = ax.imshow(data, origin="lower", vmin=0.5, vmax=1)
fig.colorbar(p)

ax.set_title(waveform)
# fig.savefig(method + ".png")
plt.show()
