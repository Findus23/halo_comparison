from numba import njit

gamma = 5 / 3
hydro_gamma_minus_one = gamma - 1
const_primordial_He_fraction_cgs = 0.248
hydrogen_mass_function = 1 - const_primordial_He_fraction_cgs
mu_neutral = 4.0 / (1.0 + 3.0 * hydrogen_mass_function)
mu_ionised = 4.0 / (8.0 - 5.0 * (1.0 - hydrogen_mass_function))
T_transition = 1.0e4
UnitMass_in_cgs = 1.98848e43  # 10^10 M_sun in grams
UnitLength_in_cgs = 3.08567758e24  # 1 Mpc in centimeters
UnitVelocity_in_cgs = 1e5  # 1 km/s in centimeters per second
UnitTime_in_cgs = UnitLength_in_cgs / UnitVelocity_in_cgs
const_proton_mass_cgs = 1.67262192369e-24
const_boltzmann_k_cgs = 1.380649e-16

const_proton_mass = const_proton_mass_cgs / UnitMass_in_cgs
const_boltzmann_k = const_boltzmann_k_cgs / UnitMass_in_cgs / UnitLength_in_cgs ** 2 * (UnitTime_in_cgs ** 2)


@njit
def calculate_T(u):
    T_over_mu = (
            hydro_gamma_minus_one * u * const_proton_mass / const_boltzmann_k
    )
    if T_over_mu > (T_transition + 1) / mu_ionised:
        return T_over_mu / mu_ionised
    elif T_over_mu < (T_transition - 1) / mu_neutral:
        return T_over_mu / mu_neutral
    else:
        return T_transition


if __name__ == "__main__":
    print(const_proton_mass)
    print(const_boltzmann_k)
    print()
    internal_energies = [6.3726251e+02, 7.7903375e+02, 1.7425287e+04, 6.4113910e+04, 3.8831848e+04,
                         1.1073163e+03, 7.7394878e+03, 7.5230023e+04, 9.1036992e+04, 2.4060946e+00]

    for u in internal_energies:
        print(calculate_T(u))
