import numpy as np

from paths import base_dir
from read_vr_files import read_velo_halos

waveform = "shannon"
resolution = 128

dir = base_dir / f"{waveform}_{resolution}_100"
halos = read_velo_halos(dir)

vmax = []

for _, halo in halos.iterrows():
    if 40 < halo["npart"] < 60:
        vmax.append(halo["Vmax"])

vmax = np.array(vmax)
print(vmax.mean())
print(vmax.std())
