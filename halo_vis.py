from sys import argv
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cic import cic_from_radius
from paths import base_dir, vis_datafile, has_1024_simulations
from read_vr_files import read_velo_halo_particles
from readfiles import read_file
from utils import waveforms

all_in_area = True

Coords = Tuple[float, float, float, float]  # radius, X, Y, Z


def load_halo_data(waveform: str, resolution: int, halo_id: int, coords: Coords):
    dir = base_dir / f"{waveform}_{resolution}_100"
    df, meta = read_file(dir / "output_0004.hdf5")
    df_halo, halo_lookup, unbound = read_velo_halo_particles(dir, skip_halo_particle_ids=all_in_area)

    halo = df_halo.loc[halo_id]
    if coords:
        radius, X, Y, Z = coords
    else:
        radius = halo["R_size"]
        X = halo["Xc"]
        Y = halo["Yc"]
        Z = halo["Zc"]
        coords: Coords = radius, X, Y, Z
    if all_in_area:
        df = df[df["X"].between(X - radius, X + radius)]
        df = df[df["Y"].between(Y - radius, Y + radius)]
        halo_particles = df[df["Z"].between(Z - radius, Z + radius)]
    else:
        halo_particle_ids = halo_lookup[halo_id]
        del halo_lookup
        del unbound
        halo_particles = df.loc[list(halo_particle_ids)]
    return halo, halo_particles, meta, coords


def get_comp_id(ref_waveform: str, reference_resolution: int, comp_waveform: str, comp_resolution: int):
    return f"{ref_waveform}_{reference_resolution}_100_{comp_waveform}_{comp_resolution}_100_velo.csv"


def map_halo_id(halo_id: int, ref_waveform: str, reference_resolution: int, comp_waveform: str, comp_resolution: int):
    file = base_dir / "comparisons" / get_comp_id(ref_waveform, reference_resolution, comp_waveform, comp_resolution)
    print("opening", file)
    df = pd.read_csv(file)
    mapping = {}
    for index, line in df.iterrows():
        mapping[int(line["ref_ID"])] = int(line["comp_ID"])
    return mapping[halo_id]


def imsave(rho, file_name: str):
    # ax.scatter(Xs, Ys)
    # plt.show()
    cmap = plt.cm.viridis
    data = np.log(1.001 + rho)
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    image = cmap(norm(data))
    print(file_name)
    plt.imsave(file_name, image)


def main():
    resolutions = [128, 256, 512]
    if has_1024_simulations:
        resolutions.append(1024)
    initial_halo_id = int(argv[1])
    first_halo = True
    rhos = {}
    ref_waveform = "shannon"
    ref_resolution = 128
    coords = {}
    for wf in waveforms:
        coords[wf] = None
    vmin = np.Inf
    vmax = -np.Inf
    with h5py.File(vis_datafile, "a") as vis_out:
        halo_group = vis_out.create_group(str(initial_halo_id))
        for waveform in waveforms:
            for resolution in resolutions:
                if first_halo:
                    assert ref_resolution == resolution
                    assert ref_waveform == waveform
                    halo_id = initial_halo_id
                    first_halo = False
                else:
                    halo_id = map_halo_id(initial_halo_id, ref_waveform, ref_resolution, waveform, resolution)
                halo, halo_particles, meta, image_coords = load_halo_data(waveform, resolution, halo_id,
                                                                          coords[waveform])
                if not coords[waveform]:
                    coords[waveform] = image_coords
                print(coords[waveform])
                print("mass", halo["Mvir"])
                # print("sleep")
                # sleep(100)
                radius, X, Y, Z = coords[waveform]
                rho, _ = cic_from_radius(
                    halo_particles.X.to_numpy(), halo_particles.Y.to_numpy(),
                    1000, X, Y, radius, periodic=False)
                rhos[(waveform, resolution)] = rho
                vmin = min(rho.min(), vmin)
                vmax = max(rho.max(), vmax)
                dataset_group = halo_group.create_group(f"{waveform}_{resolution}")
                dataset_group.create_dataset("rho", data=rho, compression='gzip', compression_opts=5)
                dataset_group.create_dataset("coords", data=coords[waveform])
                dataset_group.create_dataset("mass", data=meta.particle_mass)
                dataset_group.create_dataset("halo_id", data=halo_id)
                imsave(rho, f"out_halo{initial_halo_id}_{waveform}_{resolution}_{halo_id}.png")
        halo_group.create_dataset("vmin_vmax", data=[vmin, vmax])


if __name__ == '__main__':
    main()
