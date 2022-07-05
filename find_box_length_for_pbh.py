# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:19:55 2022

@author: Ben Melville
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Cosmological parameters:
n_s = 0.97
Omega_M_0 = 0.3099
Omega_Lambda_0 = 0.690021
Omega_R_0 = 0.0
h = 0.67742
R = 40  # for R=40, sigma is only 0.113, R>40 yields nan in one integration
z = 0
prop_factor = 289.98204077151246  # found from running R=8, z=0, requiring that sigma=0.8 should hold

# L_box / N_res = 0.009235859375        #for 10^5 M_sun particle mass -> 4096 almost enough to go to z=0
# L_box / N_res = 0.019897945879081797  #for 10^6 M_sun particle mass -> 2048 just enough to go to z=0


def E_squared(z, Omega_Lambda_0, Omega_M_0, Omega_R_0):
    Omega_0 = Omega_Lambda_0 + Omega_M_0 + Omega_R_0
    return (
        Omega_Lambda_0
        + (1 - Omega_0) * (1 + z) ** 2
        + Omega_M_0 * (1 + z) ** 3
        + Omega_R_0 * (1 + z) ** 4
    )


def Omega_M(z, Omega_Lambda_0, Omega_M_0, Omega_R_0):
    return Omega_M_0 * (1 + z) ** 3 / E_squared(z, Omega_Lambda_0, Omega_M_0, Omega_R_0)


def Omega_Lambda(z, Omega_Lambda_0, Omega_M_0, Omega_R_0):
    return Omega_Lambda_0 / E_squared(z, Omega_Lambda_0, Omega_M_0, Omega_R_0)


def g(z, Omega_Lambda_0, Omega_M_0, Omega_R_0):
    Omega_M_local = Omega_M(z, Omega_Lambda_0, Omega_M_0, Omega_R_0)
    Omega_Lambda_local = Omega_Lambda(z, Omega_Lambda_0, Omega_M_0, Omega_R_0)
    return (
        5
        * Omega_M_local
        / (
            2
            * (
                Omega_M_local ** (4 / 7)
                - Omega_Lambda_local
                + (1 + Omega_M_local / 2) * (1 + Omega_Lambda_local / 70)
            )
        )
    )


def D(z, Omega_Lambda_0, Omega_M_0, Omega_R_0):
    return g(z, Omega_Lambda_0, Omega_M_0, Omega_R_0) / (1 + z)


def keq(Omega_M_0, h):
    return Omega_M_0 * h ** 2


def T(k, k_eq):
    if k < k_eq:
        return 1
    elif k >= k_eq:
        return np.log(k / k_eq) / (k / k_eq) ** 2


def P_i(k, n_s):
    return k ** n_s


def P(k, z, n_s, Omega_M_0, Omega_Lambda_0, Omega_R_0, h):
    k_eq = keq(Omega_M_0, h)
    return (
        P_i(k, n_s) * T(k, k_eq) ** 2 * D(z, Omega_Lambda_0, Omega_M_0, Omega_R_0) ** 2
    )


def W_R(k, R):
    argument = k * R
    return 3 / argument ** 2 * (np.sin(argument) - argument * np.cos(argument))


def integrand(k, z, R, n_s, Omega_M_0, Omega_Lambda_0, Omega_R_0, h):
    return P(k, z, n_s, Omega_M_0, Omega_Lambda_0, Omega_R_0, h) * W_R(k, R) * k ** 2


k_lower = 0.0
k_eq = keq(Omega_M_0, h)
k_higher = 100 * k_eq

integral, error = integrate.quad(
    integrand,
    k_lower,
    k_eq,
    args=(z, R, n_s, Omega_M_0, Omega_Lambda_0, Omega_R_0, h),
    limit=100,
)
integral_2, error_2 = integrate.quad(
    integrand,
    k_eq,
    k_higher,
    args=(z, R, n_s, Omega_M_0, Omega_Lambda_0, Omega_R_0, h),
    limit=100,
)

sigma_squared = (integral + integral_2) / (2 * np.pi ** 2)
sigma = np.sqrt(sigma_squared) * prop_factor

print(sigma)


# k_plot = np.logspace(np.log10(k_lower), np.log10(k_eq), 1000)
# T_plot = [T(k, k_eq) for k in k_plot]

# k_plot_2 = np.logspace(np.log10(k_eq), np.log10(k_higher), 1000)
# T_plot_2 = [T(k, k_eq) for k in k_plot_2]

# plt.title("Linear transfer function")
# plt.xlabel("log(k)")
# plt.ylabel("log(T(k))")
# plt.loglog(k_plot, T_plot, ".", markersize=5)
# plt.loglog(k_plot_2, T_plot_2, ".", markersize=5)
