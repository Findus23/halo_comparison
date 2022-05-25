import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cic import cic_from_radius
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
    del halo_lookup
    del unbound
    halo_particles = df.loc[list(halo_particle_ids)]
    return halo, halo_particles, meta


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
    initial_halo_id = 87
    first_halo = True
    rhos = {}
    ref_waveform = "shannon"
    ref_resolution = 128
    radius = None
    vmin = np.Inf
    vmax = -np.Inf
    with h5py.File("vis.cache.hdf5", "w") as vis_out:
        for waveform in ["shannon", "DB2", "DB8"]:
            for resolution in [128, 256, 512]:
                if first_halo:
                    assert ref_resolution == resolution
                    assert ref_waveform == waveform
                    halo_id = initial_halo_id
                    first_halo = False
                else:
                    halo_id = map_halo_id(initial_halo_id, ref_waveform, ref_resolution, waveform, resolution)

                halo, halo_particles, meta = load_halo_data(waveform, resolution, halo_id=halo_id)
                # print("sleep")
                # sleep(100)
                if not radius:
                    radius = halo["R_size"]
                    X = halo["Xc"]
                    Y = halo["Yc"]
                rho, extent = cic_from_radius(
                    halo_particles.X.to_numpy(), halo_particles.Y.to_numpy(),
                    1000, X, Y, radius)
                rhos[(waveform, resolution)] = rho
                vmin = min(rho.min(), vmin)
                vmax = max(rho.max(), vmax)
                vis_out.create_dataset(f"{waveform}_{resolution}_rho", data=rho, compression='gzip', compression_opts=5)
                vis_out.create_dataset(f"{waveform}_{resolution}_extent", data=extent)
                vis_out.create_dataset(f"{waveform}_{resolution}_mass", data=meta.particle_mass)
                imsave(rho, f"out_halo{initial_halo_id}_{waveform}_{resolution}_{halo_id}.png")
        vis_out.create_dataset("vmin_vmax", data=[vmin, vmax])


if __name__ == '__main__':
    main()
