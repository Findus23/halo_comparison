import random
from pathlib import Path
from sys import argv
from time import sleep

import matplotlib.pyplot as plt

from nfw import fit_nfw
from read_vr_files import read_velo_profiles

bin_edges, mass_profiles, density_profiles = read_velo_profiles(Path(argv[1]))

with open("ids.txt") as f:
    text = f.read()
    ids = {int(id) for id in text.split(",")}
    # print(ids)

for i, profile in enumerate(density_profiles):
    is_odd_halo = i in ids
    if not is_odd_halo:
        if random.random() > 0.1:
            continue
    else:
        popt = fit_nfw(bin_edges, profile)
        print(popt)
        sleep(1)
    color = "red" if is_odd_halo else "lightgray"

    plt.loglog(bin_edges, profile, color=color, alpha=0.1)
plt.show()
