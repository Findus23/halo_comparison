# Call with spectra_computation.py <time> <kind>
# time = 'ics' for ICs, = 'end' for final results
# kind = 'power' for power spectra comparing same resolution, 'cross' for comparing across all resolutions

import itertools
import subprocess
from multiprocessing import Pool, cpu_count
from sys import argv

from paths import base_dir, spectra_dir


def run_spectra(waveform: str, resolution_1: int, resolution_2: int, Lbox: int, time: str):
    print("starting")
    setup_1 = f'{waveform}_{resolution_1}_{Lbox}'
    setup_2 = f'{waveform}_{resolution_2}_{Lbox}'

    # #For ICs: time == 'ics'
    if time == 'ics':
        subprocess.run([
            str(spectra),
            '--ngrid',
            '1024',
            '--format=4',  # This seems to work, but is not as readable
            '--output',
            str(base_dir / f'spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_ics_{resolution_1}_{resolution_2}'),
            '--input',
            str(base_dir / f'{setup_1}/ics_{setup_1}.hdf5'),
            '--input',
            str(base_dir / f'{setup_2}/ics_{setup_2}.hdf5')
        ], check=True)
        print("end")

    # #For evaluation of final results: time == 'end'
    elif time == 'end':
        subprocess.run([
            str(spectra),
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
    else:
        raise ValueError(f"invalid time ({time})")


def power_run(waveforms: list, resolutions: list, Lbox: int, time: str):
    args = []
    for waveform in waveforms:
        for resolution in resolutions:
            args.append((
                waveform,
                resolution,
                resolution,
                Lbox,
                time
            ))
    return args


def cross_run(waveforms: list, resolutions: list, Lbox: int, time: str):
    args = []
    for waveform in waveforms:
        for res1, res2 in itertools.combinations(resolutions, 2):
            args.append({
                "waveform": waveform,
                "resolution_1": res1,
                "resolution_2": res2,
                "Lbox": Lbox,
                "time": time
            })
    return args


if __name__ == '__main__':
    Lbox = 100
    k0 = 2 * 3.14159265358979323846264338327950 / Lbox
    resolutions = [128, 256, 512]
    waveforms = ["DB2", "DB4", "DB8", "shannon"]

    spectra = spectra_dir / 'spectra'
    time = argv[1]

    if argv[2] == 'power':
        args = power_run(waveforms=waveforms, resolutions=resolutions, Lbox=Lbox, time=time)

    elif argv[2] == 'cross':
        args = cross_run(waveforms=waveforms, resolutions=resolutions, Lbox=Lbox, time=time)
    else:
        raise ValueError("missing argv[2] (power|cross)")
    with Pool(processes=3) as p:
        p.starmap(run_spectra, args)
