# import numpy as np
# import matplotlib.pyplot as plt
# import itertools
# import pandas as pd
import subprocess

from paths import base_dir, spectra_dir

Lbox = 100
k0 = 2 * 3.14159265358979323846264338327950 / Lbox
resolutions = [128, 256, 512] #leave out 512 for now, cause this might only work at the VSC

spectra = spectra_dir / 'spectra'

for k, waveform in enumerate(["DB2", "DB4", "DB8", "shannon"]):

    for l, resolution in enumerate(resolutions):
        setup = f'{waveform}_{resolution}_{Lbox}'
        dir = base_dir / setup
        subprocess.run([str(spectra),
        '--ngrid',
        # f'{2 * resolution}',
        '1024',
        # #For evaluation of ICs
        '--format=4', #This seems to work, but is not as readible
        '--output',
        str(base_dir / f'spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_ics_{resolution}_{resolution}'),
        '--input',
        str(base_dir / f'{setup}/ics_{setup}.hdf5'),
        '--input',
        str(base_dir / f'{setup}/ics_{setup}.hdf5')
        
        # #For evaluation of final results
        # '--format=3', 
        # '--output',
        # str(base_dir / f'spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a4_{resolution}_{resolution}'),
        # '--input',
        # str(base_dir / f'{setup}/output_0004.hdf5'),
        # '--input',
        # str(base_dir / f'{setup}/output_0004.hdf5')
        ], check=True)
        break
    break

