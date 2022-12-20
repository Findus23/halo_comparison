# Call with spectra_computation.py <time> <kind>
# time = 'ics' for ICs, = 'z=1' for redshift z=1, = 'end' for final results
# kind = 'power' for power spectra comparing same resolution, 'cross' for comparing across all resolutions

import itertools
import subprocess
from multiprocessing import Pool
from sys import argv

from jobrun.jobrun import jobrun
from paths import base_dir, spectra_dir
from spectra_plot import waveforms

vsc = True


def spectra_jobrun(args):
    if vsc:
        jobrun(args, time="12:00:00", tasks=128, mem=128)
    else:
        subprocess.run(args, check=True)


def run_spectra(
        waveform: str, resolution_1: int, resolution_2: int, Lbox: int, time: str
):
    print("starting")
    setup_1 = f"{waveform}_{resolution_1}_{Lbox}"
    setup_2 = f"{waveform}_{resolution_2}_{Lbox}"

    # #For ICs: time == 'ics'
    if time == "ics":
        output_file = (
                base_dir
                / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_ics_{resolution_1}_{resolution_2}_cross_spectrum.txt"
        )
        if output_file.exists():
            print(f"{output_file} already exists, skipping.")
            return
        spectra_jobrun(
            [
                str(spectra),
                "--ngrid",
                "2048",
                "--format=4",  # This seems to work, but is not as readable
                "--output",
                str(
                    base_dir
                    / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_ics_{resolution_1}_{resolution_2}"
                ),
                "--input",
                str(base_dir / f"{setup_1}/ics_{setup_1}.hdf5"),
                "--input",
                str(base_dir / f"{setup_2}/ics_{setup_2}.hdf5"),
            ]
        )

    # #For evaluation of results at redshift z=1: time == 'z=1' | NOT ADAPTED FOR VSC5 YET!
    elif time == "z=1":
        output_file = (
                base_dir
                / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a2_{resolution_1}_{resolution_2}_cross_spectrum.txt"
        )
        if output_file.exists():
            print(f"{output_file} already exists, skipping.")
            return
        spectra_jobrun(
            [
                str(spectra),
                "--ngrid",
                "1024",
                "--format=3",
                "--output",
                str(
                    base_dir
                    / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a2_{resolution_1}_{resolution_2}"
                ),
                "--input",
                str(base_dir / f"{setup_1}/output_0002.hdf5"),
                "--input",
                str(base_dir / f"{setup_2}/output_0002.hdf5"),
            ]
        )


    # #For evaluation of final results: time == 'end'
    elif time == "end":
        output_file = (
                base_dir
                / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a4_{resolution_1}_{resolution_2}_cross_spectrum.txt"
        )
        if output_file.exists():
            print(f"{output_file} already exists, skipping.")
            return
        spectra_jobrun(
            [
                str(spectra),
                "--ngrid",
                "2048",
                "--format=3",
                "--output",
                str(
                    base_dir
                    / f"spectra/{waveform}_{Lbox}/{waveform}_{Lbox}_a4_{resolution_1}_{resolution_2}"
                ),
                "--input",
                str(base_dir / f"{setup_1}/output_0004.hdf5"),
                "--input",
                str(base_dir / f"{setup_2}/output_0004.hdf5"),
            ]
        )
    else:
        raise ValueError(f"invalid time ({time})")

    print("end")


def power_run(resolutions: list, Lbox: int, time: str):
    args = []
    for waveform in waveforms:
        for resolution in resolutions:
            args.append((waveform, resolution, resolution, Lbox, time))
    return args


def cross_run(resolutions: list, Lbox: int, time: str):
    args = []
    for waveform in waveforms:
        for res1, res2 in itertools.combinations(resolutions, 2):
            args.append((waveform, res1, res2, Lbox, time))
    return args


if __name__ == "__main__":
    #    input("are you sure you want to run this? This might need a large amount of memory")
    Lbox = 100
    resolutions = [128, 256, 512, 1024]

    spectra = spectra_dir / "spectra"
    time = argv[1]

    if argv[2] == "power":
        args = power_run(resolutions=resolutions, Lbox=Lbox, time=time)

    elif argv[2] == "cross":
        args = cross_run(resolutions=resolutions, Lbox=Lbox, time=time)
    else:
        raise ValueError("missing argv[2] (power|cross)")
    with Pool(processes=1) as p:
        p.starmap(run_spectra, args)
