from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from cic import cic_from_radius
from cumulative_mass_profiles import cumulative_mass_profile
from paths import auriga_dir
from readfiles import read_file, read_halo_file

softening_length = 0.026041666666666668

fig1: Figure = plt.figure(figsize=(9, 6))
ax1: Axes = fig1.gca()
fig2: Figure = plt.figure(figsize=(9, 6))
ax2: Axes = fig2.gca()

for ax in [ax1,ax2]:
    ax.set_xlabel(r'R / R$_\mathrm{group}$')
ax1.set_ylabel(r'M [$10^{10} \mathrm{M}_\odot$]')
ax2.set_ylabel("density [$\\frac{10^{10} \\mathrm{M}_\\odot}{Mpc^3}$]")


@dataclass
class Result:
    title: str
    rho: np.ndarray


images = []
vmin = np.Inf
vmax = -np.Inf

for i, dir in enumerate(auriga_dir.glob("*")):
    if "auriga" not in dir.name:
        continue
    Xc_adrian = 56.50153741810241
    Yc_adrian = 49.40761085700951
    Zc_adrian = 49.634393647291695
    Xc = 58.25576087992683
    Yc = 51.34632916228137
    Zc = 51.68749302578122

    is_by_adrian = "arj" in dir.name
    if not dir.is_dir():
        continue
    input_file = dir / "output_0007.hdf5"
    if is_by_adrian:
        input_file = dir / "output_0000.hdf5"
    print(input_file)
    df, particles_meta = read_file(input_file)
    df_halos = read_halo_file(input_file.with_name("fof_" + input_file.name))

    # particles_in_halo = df.loc[df["FOFGroupIDs"] == 3]

    halo_id = 1
    while True:
        particles_in_halo = df.loc[df["FOFGroupIDs"] == halo_id]
        if len(particles_in_halo) > 1:
            break
        halo_id += 1

    halo = df_halos.loc[halo_id]
    log_radial_bins, bin_masses, bin_densities, group_radius = cumulative_mass_profile(
        df, halo, particles_meta, plot=False
    )
    ax1.loglog(log_radial_bins[:-1], bin_masses, label=str(dir.name), c=f"C{i}")

    ax2.loglog(log_radial_bins[:-1], bin_densities, label=str(dir.name), c=f"C{i}")

    for ax in [ax1,ax2]:
        ax.axvline(4 * softening_length, color=f"C{i}", linestyle="dotted")

    X, Y, Z = df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy()
    print()
    print(Yc - Yc_adrian)
    # shift: (-6, 0, -12)
    if not is_by_adrian:
        xshift = Xc - Xc_adrian
        yshift = Yc - Yc_adrian
        zshift = Zc - Zc_adrian
        print("shift", xshift, yshift, zshift)

        X -= 1.9312
        Y -= 1.7375
        Z -= 1.8978

    rho, extent = cic_from_radius(X, Z, 500, Xc_adrian, Yc_adrian, 5, periodic=False)

    vmin = min(vmin, rho.min())
    vmax = max(vmax, rho.max())

    images.append(Result(
        rho=rho,
        title=str(dir.name)
    ))

    # plot_cic(
    #     rho, extent,
    #     title=str(dir.name)
    # )
ax1.legend()
ax2.legend()

fig3: Figure = plt.figure(figsize=(9, 6))
axes: List[Axes] = fig3.subplots(2, 3, sharex=True, sharey=True).flatten()

for result, ax in zip(images, axes):
    data = 1.1 + result.rho
    vmin_scaled = 1.1 + vmin
    vmax_scaled = 1.1 + vmax
    img = ax.imshow(data.T, norm=LogNorm(vmin=vmin_scaled, vmax=vmax_scaled), extent=extent,
                     origin="lower")
    ax.set_title(result.title)

fig3.tight_layout()
fig3.subplots_adjust(right=0.825)
cbar_ax = fig3.add_axes([0.85, 0.15, 0.05, 0.7])
fig3.colorbar(img, cax=cbar_ax)

fig1.savefig(Path("~/tmp/auriga1.pdf").expanduser())
fig2.savefig(Path("~/tmp/auriga2.pdf").expanduser())
fig3.savefig(Path("~/tmp/auriga3.pdf").expanduser())

plt.show()
