import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cic import cic_from_radius, plot_cic
from find_center import find_center
from halo_mass_profile import halo_mass_profile
from paths import pbh_dir
from readfiles import read_g4_file, ParticlesMeta
from utils import figsize_from_page_fraction


def cic_comparison(pbh_high_coord, ref_high_coord,center):
    rhos = []
    i = 0
    for coord in [ref_high_coord, pbh_high_coord]:
        rho, extent = cic_from_radius(
            coord[::, 0], coord[::, 1], 3000, center[0], center[1], 2, periodic=False
        )
        rhos.append(rho)

        plot_cic(
            rho, extent, title=f"test{i}"
        )
        i += 1
    plot_cic(
        np.absolute(rhos[0] - rhos[1]), extent, title="abs(diff)"
    )


def main():
    ref_data = read_g4_file(
        pbh_dir / "cdm" / "snapshot_005.hdf5",
        zoom_type="cdm")
    pbh_data = read_g4_file(
        pbh_dir / "10000sigma" / "snapshot_005.hdf5",
        zoom_type="pbh")
    center = [30, 32, 30]
    cic_comparison(ref_data[0], pbh_data[0],center)
    fig1: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax1: Axes = fig1.gca()
    fig2: Figure = plt.figure(figsize=figsize_from_page_fraction())
    ax2: Axes = fig2.gca()
    centered = False
    for data in [ref_data, pbh_data]:
        highres_coords, lowres_coords, highres_mass, lowres_mass = data
        df = pd.DataFrame(highres_coords, columns=["X", "Y", "Z"])
        particles_meta = ParticlesMeta(particle_mass=highres_mass)
        # center = np.median(highres_coords, axis=0)
        print(center)
        # if not centered:
        #     center = find_center(df, center, initial_radius=0.01)
        centered = True
        log_radial_bins, bin_masses, bin_densities, center = halo_mass_profile(
            df, center, particles_meta, plot=False, num_bins=100, vmin=0.002, vmax=5
        )
        ax1.loglog(log_radial_bins[:-1], bin_masses)

        ax2.loglog(log_radial_bins[:-1], bin_densities)
    plt.show()
    cic_comparison(ref_data[0], pbh_data[0],center)


if __name__ == '__main__':
    main()
