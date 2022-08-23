import numpy as np

hubbleparam = 0.6777
omega0 = 0.307
omegab = 0.04825
Gconst = 43.0187 
zstart = 127
astart = 1 / (1 + zstart)

m_gas = 0.00310523 / hubbleparam
m_highres = 0.0166524 / hubbleparam
m_lowres_1 = 0.15806103 / hubbleparam
m_lowres_2 = 1.2644882 / hubbleparam

n_gas = 174432
n_highres = 174432
n_lowres_1 = 46964
n_lowres_2 = 2135520 - n_lowres_1

boxsize = 67.77 / hubbleparam

Hsquared = (hubbleparam * 100) ** 2 * (omega0 / (astart) ** 3 + 1 - omega0)

rhocrit = 3 * (hubbleparam * 100) ** 2 / (8 * np.pi * Gconst)
print(rhocrit / hubbleparam ** 2)

#m_gas = omegab * 
mass_b = m_gas * n_gas
omega_b = mass_b / (boxsize ** 3) / (3 * (hubbleparam * 100) ** 2 / (8 * np.pi * Gconst))

mass_lowres_1 = m_lowres_1 * n_lowres_1
mass_lowres_2 = m_lowres_2 * n_lowres_2
mass_lowres = mass_lowres_1 + mass_lowres_2
mass_highres = m_highres * n_highres
mass_total = mass_b + mass_highres + mass_lowres

omega_0 = mass_total / (boxsize ** 3) / (3 * (hubbleparam * 100) ** 2 / (8 * np.pi * Gconst))


print(omega_b)
print(omega_0)

print(mass_b * hubbleparam)