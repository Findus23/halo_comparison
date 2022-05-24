import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from pyvista import Axes

from cic import cic_deposit
from paths import base_dir
from read_vr_files import read_velo_halos
from readfiles import read_file

show_unbound = False


def load_halo_data(waveform: str, resolution: int, halo_id: int):
    dir = base_dir / f"{waveform}_{resolution}_100"
    df, meta = read_file(dir)
    df_halo, halo_lookup, unbound = read_velo_halos(dir, recursivly=False, skip_unbound=not show_unbound)
    if show_unbound:
        for k, v in halo_lookup.items():
            v.update(unbound[k])

    halo = df_halo.loc[halo_id]
    halo_particle_ids = halo_lookup[halo_id]
    halo_particles = df.loc[list(halo_particle_ids)]
    return halo, halo_particles


def calc_cic(halo, halo_particles, diameter, X, Y):
    Xs = (halo_particles.X.to_numpy() - X) / diameter / 2 + 0.5
    Ys = (halo_particles.Y.to_numpy() - Y) / diameter / 2 + 0.5
    print(min(Xs), max(Xs))
    rho = cic_deposit(Xs, Ys, 1000)
    return rho


def get_comp_id(ref_waveform: str, reference_resolution: int, comp_waveform: str, comp_resolution: int):
    return f"{ref_waveform}_{reference_resolution}_100_{comp_waveform}_{comp_resolution}_100_velo.csv"


def map_halo_id(halo_id: int, ref_waveform: str, reference_resolution: int, comp_waveform: str, comp_resolution: int):
    file = base_dir / "comparisons" / get_comp_id(ref_waveform, reference_resolution, comp_waveform, comp_resolution)
    print("opening", file)
    df = pd.read_csv(file)
    mapping = {}
    for index, line in df.iterrows():
        mapping[int(line["ref_ID"])] = int(line["comp_ID"])
    print(mapping)
    return mapping[halo_id]


def plot_halo(rho, file_name: str):
    # ax.scatter(Xs, Ys)
    # plt.show()
    cmap = plt.cm.viridis
    data = np.log(1.001 + rho)
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    image = cmap(norm(data))
    print(file_name)
    plt.imsave(file_name, image)
    # fig: Figure = plt.figure()
    # ax: Axes = fig.gca()
    # i = ax.imshow(1.001 + rho, norm=LogNorm())
    # fig.colorbar(i)
    # plt.show()


def main():
    initial_halo_id = 87
    first_halo = True
    rhos = {}
    ref_waveform = "shannon"
    ref_resolution = 128
    diameter=None
    for waveform in ["shannon", "DB2", "DB8"]:
        for resolution in [128, 256, 512]:
            print(waveform, resolution)
            if first_halo:
                assert ref_resolution == resolution
                assert ref_waveform == waveform
                halo_id = initial_halo_id
                first_halo = False
            else:
                halo_id = map_halo_id(initial_halo_id, ref_waveform, ref_resolution, waveform, resolution)
            halo, halo_particles = load_halo_data(waveform, resolution, halo_id=halo_id)
            if not diameter:
                diameter = halo["R_size"]
                X = halo["Xc"]
                Y = halo["Yc"]
            rho = calc_cic(halo, halo_particles, diameter, X, Y)
            rhos[(waveform, resolution)] = rho
            plot_halo(rho, f"out_halo{initial_halo_id}_{waveform}_{resolution}_{halo_id}.png")


if __name__ == '__main__':
    main()
