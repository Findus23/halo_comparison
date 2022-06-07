import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from paths import base_dir

Lbox = 100
k0 = 2 * 3.14159265358979323846264338327950 / Lbox
resolution = [128, 256, 512]

# Careful: k is actually in Mpc^-1, the column is just named weirdly.
columns = ["k [Mpc]", "Pcross", "P1", "err. P1", "P2", "err. P2", "P2-1", "err. P2-1", "modes in bin"]

linestyles = ["solid", "dashed", "dotted"]
colors = ["C1", "C2", "C3", "C4"]


for k, waveform in enumerate(["DB2", "DB4", "DB8", "shannon"]):
    dir = base_dir / f'spectra/{waveform}_{Lbox}'

    for l, resolutions in enumerate(list(itertools.combinations(resolution, 2))):
        spectra_data = pd.read_csv(f'{dir}/{waveform}_{Lbox}_ics_{resolutions[0]}_{resolutions[1]}_cross_spectrum.txt', sep=" ", skipinitialspace=True, header=None, names=columns, skiprows=1)

        #only consider rows above resolution limit
        spectra_data = spectra_data[spectra_data["k [Mpc]"] >= k0]

        k = spectra_data["k [Mpc]"]
        p1 = spectra_data["P1"]
        p1_error = spectra_data["err. P1"]
        p2 = spectra_data["P2"]
        p2_error = spectra_data["err. P2"]

