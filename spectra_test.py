import numpy as np
import pandas as pd
import scipy.special as sf
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def read_monofonic_file(file: str):
    return pd.read_csv(
        file,
        sep=" ",
        skipinitialspace=True,
        header=None,
        names=[
            "k", "P_dtot", "P_dcdm", "P_dbar", "P_tcdm", "P_tbar", "..", "...", "....", "....."
        ],
        skiprows=1,
    )


def read_spectra_file(file: str):
    return pd.read_csv(
        # f"/home/lukas/cosmos_data/monofonic_tests/spectra/DB8_100/DB8_100_ics_256_256_cross_spectrum.txt",
        file,
        sep=" ",
        skipinitialspace=True,
        header=None,
        names=[
            "k",
            "Pcross",
            "P1",
            "err. P1",
            "P2",
            "err. P2",
            "P2-1",
            "err. P2-1",
            "modes in bin",
        ],
        skiprows=1,
    )


monofonic_spectra = read_monofonic_file(
    f"/home/lukas/cosmoca/cosmoca-reproducibility/monofonic_tests/input_powerspec.txt")
print(monofonic_spectra)


def Dplus(lambda0, a):
    return a * np.sqrt(1.0 + lambda0 * a ** 3) * \
        sf.hyp2f1(3.0 / 2.0, 5.0 / 6.0, 11.0 / 6.0, -lambda0 * a ** 3)


OmegaLambda = 0.6901
OmegaM = 0.3099
lambda_0 = OmegaLambda / OmegaM
a_ini = 0.02
factor = Dplus(lambda_0, a_ini) ** 2 / Dplus(lambda_0, 1) ** 2

sim_spectra = read_spectra_file(
    "/home/lukas/cosmoca/cosmoca-reproducibility/monofonic_tests/out.txt_cross_spectrum.txt")
sim_spectra_ics_new = read_spectra_file(
    "/home/lukas/cosmoca/cosmoca-reproducibility/monofonic_tests/DB8_spectra.txt_cross_spectrum.txt")
sim_spectra_ics_old = read_spectra_file(
    "/home/lukas/cosmos_data/monofonic_tests/spectra/DB8_100/DB8_100_ics_256_256_cross_spectrum.txt", )
sim_spectra = read_spectra_file(
    "/home/lukas/cosmos_data/monofonic_tests/spectra/DB8_100/DB8_100_a4_256_256_cross_spectrum.txt", )
sim_spectra_new = read_spectra_file(
    "/home/lukas/cosmoca/cosmoca-reproducibility/monofonic_tests/DB8_spectra_end_cross_spectrum.txt")

fig: Figure = plt.figure()
ax: Axes = fig.gca(

)
ax.loglog(sim_spectra.k, sim_spectra.P1*factor, label="sim")
ax.loglog(sim_spectra_new.k, sim_spectra_new.P1*factor, label="sim new")
ax.loglog(sim_spectra_ics_old.k, sim_spectra_ics_old.P1, label="ics old")
ax.loglog(sim_spectra_ics_new.k, sim_spectra_ics_new.P1, label="ics new")
ax.loglog(monofonic_spectra.k, monofonic_spectra.P_dtot, linestyle="dotted", label="ref")
# ax.loglog(monofonic_spectra.k, monofonic_spectra.P_dcdm, label="P_dcdm")
# ax.loglog(monofonic_spectra.k, monofonic_spectra.P_dbar, label="P_dbar")
plt.legend()
plt.show()
