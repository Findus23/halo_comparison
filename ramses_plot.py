from pathlib import Path
from subprocess import run
from typing import List

import numpy as np
import pynbody
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pynbody.array import SimArray
from pynbody.snapshot import FamilySubSnap
from pynbody.snapshot.ramses import RamsesSnap
from scipy import constants

ramses_imager = Path("~/cosmoca/RamsesImager/build/RamsesImager").expanduser()
dir = Path("/media/ssd/cosmos_data/coxeter/richings21_ics/richings21_bary_ramses_7_10_11/")


def get_slice_argument(
        extent: List[float], zcenter: float,
        ramses_dir: Path, interpolation_method: str,
        depth: float):
    xmin, xmax, ymin, ymax = extent
    # interpolate = interpolation_method == "linear"
    interpolate = False
    arguments = {
        "x": (xmin + xmax) / 2,
        "y": (ymin + ymax) / 2,
        "z": zcenter,
        "w": xmax - xmin,
        "h": ymax - ymin,
        "d": depth,
        "l": 14
    }
    args = [str(ramses_imager)]
    for k, v in arguments.items():
        args.append(f"-{k} {v}")

    if interpolate:
        args.append("-i")

    args.append(str(ramses_dir / "info_00009.txt"))
    return args, ramses_imager.parent


def load_slice_data(file: Path):
    with file.open("rb") as infile:
        np.fromfile(file=infile, dtype=np.int32, count=1)
        [nx, ny] = np.fromfile(file=infile, dtype=np.int32, count=2)
        np.fromfile(file=infile, dtype=np.int32, count=1)

        np.fromfile(file=infile, dtype=np.int32, count=1)
        data: np.ndarray = np.fromfile(file=infile, dtype=np.float32, count=nx * ny)
        np.fromfile(file=infile, dtype=np.int32, count=1)
        print("NEGATIVE", (data < 0).sum())
        # np.fromfile(file=infile, dtype=np.int32, count=1)
        # cm_per_px = np.fromfile(file=infile, dtype=np.float64, count=1)[0]
        # np.fromfile(file=infile, dtype=np.int32, count=1)
    return data.reshape((nx, ny))


def main():
    file = dir / "output_00009"
    s: RamsesSnap = pynbody.load(str(file))
    gas_data: FamilySubSnap = s.gas
    h = s.properties["h"]
    a = s.properties["a"]
    print("RAMSES a", a)

    # dm
    mass_array: SimArray = s.dm["mass"]
    coord_array: SimArray = s.dm["pos"]
    masses = np.asarray(mass_array.in_units("1e10 Msol"))
    high_res_mass = np.amin(np.unique(masses))  # get lowest mass of particles

    is_high_res_particle = masses == high_res_mass

    coordinates = np.asarray(coord_array.in_units("Mpc"))
    hr_coordinates = coordinates[is_high_res_particle] / a

    np.save(f"/home/lukas/tmp/ramses_grid_hr_coordinates", hr_coordinates)
    print(high_res_mass)

    # temperature_array: SimArray = gas_data["temp"]
    # p_array: SimArray = gas_data["p"].in_units("1e10 Msol Mpc^-3 km^2 s^-2")
    # rho_array: SimArray = gas_data["rho"].in_units("1e10 Msol Mpc^-3")
    # coord_array: SimArray = gas_data["pos"].in_units("Mpc")
    # mass_array = np.asarray(gas_data["mass"].in_units("1e10 Msol"))
    # bary_coords = np.asarray(coord_array)
    # bary_properties = {
    #     "Temperatures": np.asarray(temperature_array.in_units("K")),
    #     "Pressures": np.asarray(p_array),
    #     "Densities": np.asarray(rho_array),
    #     "Entropies": np.asarray(np.log10(p_array / rho_array ** (5 / 3))),
    # }

    michael_extent = np.asarray([[32.07045664, 34.13128024],
                                 [37.562513, 39.6233366],
                                 [34.54953566, 36.61035927]])

    frac_extent = michael_extent[:2].flatten() / 100 / h
    frac_center = michael_extent[2].mean() / 100 / h
    # old extents:
    # frac_extent=[0.46938643, 0.50738643, 0.55012166, 0.58812166]
    # frac_center=0.52467408

    print("frac")

    print(frac_extent)
    print(frac_center)
    args, imager_dir = get_slice_argument(
        frac_extent, frac_center,
        file, interpolation_method="nearest",
        depth=0.00000000001
        # depth=depth
    )
    print(" ".join(args))

    run(args, cwd=imager_dir)
    property_map = {
        "Densities": "rhomap",
        "Entropies": "Smap",
        "Temperatures": "Tmap"
    }

    for property in ["cic", "Densities", "Entropies", "Temperatures"]:
        if property == "cic":
            ...
        else:

            fname = imager_dir / f"snapshot_{property_map[property]}_zproj_zobs-0p00.bin"
            grid = load_slice_data(fname).T
            if grid.sum() == 0:
                raise ValueError("all 0")
            if property == "Densities":
                # convert g/cm^3 to 1e10 Msol Mpc^-3
                solar_mass_in_gram = 1.988e33
                mpc_in_cm = constants.parsec * constants.mega * 100
                grid = np.asarray(grid)
                grid *= (mpc_in_cm ** 3 / solar_mass_in_gram / 1e10)
            print(grid.shape, "shape")
            print(grid)
            plt.figure()
            plt.imshow(
                grid,
                norm=LogNorm(),
                interpolation="none",
                origin="lower",
                extent=frac_extent * 100,
            )
            plt.colorbar()
            plt.savefig(f"/home/lukas/tmp/ramses_grid_{property}.pdf")

            np.save(f"/home/lukas/tmp/ramses_grid_{property}", grid)
    plt.show()


if __name__ == '__main__':
    main()
