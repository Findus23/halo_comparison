from numpy import inf
from scipy.optimize import curve_fit


def nfw(r, rho_0, r_s):
    r_norm = r / r_s
    return rho_0 / (r_norm * (1 + r_norm) ** 2)


def fit_nfw(radius, densities):
    popt, pcov = curve_fit(
        nfw,
        radius,
        densities,
        verbose=1,
        method="trf",
        max_nfev=1000,
        bounds=([0, 0], [inf, 1]),
    )
    return popt
