#Call with spectra_computation.py <time> <kind> 
#time = 'ics' for ICs, = 'end' for final results
#kind = 'power' for power spectra comparing same resolution, 'cross' for comparing across all resolutions

import itertools
import re
import subprocess
from sys import argv
import wave

from paths import base_dir, spectra_dir


def spectra(waveform: str, resolution_1: int, resolution_2: int, Lbox: float, time: str):
    setup_1 = f'{waveform}_{resolution_1}_{Lbox}'
    setup_2 = f'{waveform}_{resolution_2}_{Lbox}'

    # #For ICs: time == 'ics'
    if time == 'ics':
        subprocess.run([str(spectra),
        '--ngrid',
        '1024',
        '--format=4', #This seems to work, but is not as readable
        '--output',
        str(base_dir / f'spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_ics_{resolution_1}_{resolution_2}'),
        '--input',
        str(base_dir / f'{setup_1}/ics_{setup_1}.hdf5'),
        '--input',
        str(base_dir / f'{setup_2}/ics_{setup_2}.hdf5')
        ], check=True)
    
    # #For evaluation of final results: time == 'end'
    if time == 'end':
        subprocess.run([str(spectra),
        '--ngrid',
        '1024',
        '--format=3', 
        '--output',
        str(base_dir / f'spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a4_{resolution_1}_{resolution_2}'),
        '--input',
        str(base_dir / f'{setup_1}/output_0004.hdf5'),
        '--input',
        str(base_dir / f'{setup_2}/output_0004.hdf5')
        ], check=True)

def power_run(waveforms: list, resolutions: list, Lbox: float, time: str):
    for waveform in enumerate(waveforms):
        for resolution in enumerate(resolutions):
            spectra(waveform=waveform, resolution_1=resolution, resolution_2=resolution, Lbox=Lbox, time=time)

def cross_run(waveforms: list, resolutions: list, Lbox: float, time: str):
    for waveform in enumerate(waveforms):
        for resolution_pair in itertools.combinations(resolutions, 2):
            spectra(waveform=waveform, resolution_1=resolution_pair[0], resolution_2=resolution_pair[1], Lbox=Lbox, time=time)


if __name__ == '__main__':
    Lbox = 100.0
    k0 = 2 * 3.14159265358979323846264338327950 / Lbox
    resolutions = [128, 256, 512] 
    waveforms = ["DB2", "DB4", "DB8", "shannon"]

    spectra = spectra_dir / 'spectra'
    time = argv[1]

    if argv[2] == 'power':
        power_run(waveforms=waveforms, resolutions=resolutions, Lbox=Lbox, time=time)
    
    elif argv[2] == 'cross':
        cross_run(waveforms=waveforms, resolutions=resolutions, Lbox=Lbox, time=time)

