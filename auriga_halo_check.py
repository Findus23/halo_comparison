import matplotlib.pyplot as plt
import numpy as np

from paths import auriga_dir, richings_dir, auriga_dir_new, richings_dir_new
from read_vr_files import read_velo_halos
from readfiles import read_file, read_halo_file
from rockstar import read_rockstar_halos

fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()
fig3 = plt.figure()
ax3 = fig3.gca()

for run in reversed([
    "7_8_9",
    "7_8_10",
    "7_10_10",
    "7_10_12",
    # "auriga6_halo_arj"
    "adrian_ref_new"
]):
    dir = auriga_dir_new / run
    input_file = dir / "output_0007.hdf5"
    print(dir.name)
    # df_halos = read_halo_file(input_file.with_name("fof_" + input_file.name))
    # masses = np.asarray(df_halos["Masses"])
    # if "arj" in run or "adrian" in run:
    #     input_file = dir / "output_0000.hdf5"
    # vr_halo = read_velo_halos(dir, veloname="velo_out")
    # # print(vr_halo.Structuretype)
    # # vr_halo=vr_halo[vr_halo.Structuretype!=10]
    # halo_pos = np.array([vr_halo.X, vr_halo.Y, vr_halo.Z]).T
    # print(halo_pos.shape)
    # masses = np.asarray(vr_halo["Mvir"])
    rockstar_df, _ = read_rockstar_halos(dir)
    print(rockstar_df.shape)
    halo_pos = np.array([rockstar_df.x, rockstar_df.y, rockstar_df.z]).T
    masses = np.asarray(rockstar_df.m200c) / 1e10
    # order=np.argsort(-masses)
    # halo_pos=halo_pos[order]
    # masses=masses[order]

    most_massive_halo_pos = halo_pos[0]
    diff = most_massive_halo_pos - halo_pos
    distances = np.linalg.norm(diff, axis=1)
    filter = distances < 0.095953835
    distances = distances[filter]
    masses = masses[filter]
    diff = diff[filter]
    ax1.scatter(distances, masses, label=run, s=8)
    theta = np.rad2deg(np.arccos(diff[::, 2] / distances))
    phi = np.rad2deg(np.arctan2(diff[:, 1], diff[:, 0]))
    ax2.scatter(theta, masses, label=run, s=8)
    ax3.scatter(phi, masses, label=run, s=8)
    if run == "7_10_12":
        # rvir = vr_halo.iloc[1].Rvir
        r200c = rockstar_df.iloc[1].r200c / 1000  # kpc/h to Mpc/h
        print("r200c", r200c)

    # most_massive_halo = df_halos.iloc[0]
    # most_massive_halo_pos = np.array([most_massive_halo.X, most_massive_halo.Y, most_massive_halo.Z])
    # print(df_halos.iloc[0])
    # for id, halo in df_halos.items():
    #     halo_pos = np.array([most_massive_halo.X, most_massive_halo.Y, most_massive_halo.Z])
    #     distance=np.


def torvir(x):
    return x / r200c


def fromrvir(x):
    return x * r200c


secax = ax1.secondary_xaxis('top', functions=(torvir, fromrvir))
secax.set_xlabel('$r/R200c_{main}$')
ax1.set_xlabel("r")
ax1.set_xscale("log")
ax2.set_xlabel("theta [deg]")
ax3.set_xlabel("phi [deg]")

for ax in [ax1, ax2, ax3]:
    ax.set_yscale("log")
    ax.set_ylabel("Mass")
    ax.legend()
# ax1.set_xlim(.01, 7.5)
for i, fig in enumerate([fig1, fig2, fig3]):
    fig.tight_layout()
    fig.savefig(f"/home/lukas/tmp/halo_{i}.pdf")
plt.legend()
plt.show()
