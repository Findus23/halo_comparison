import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paths import base_dir
from read_vr_files import read_velo_halos

def counts_without_inf(number_halos):
    with np.errstate(divide='ignore', invalid='ignore'):
        number_halos_inverse = np.true_divide(1, np.sqrt(number_halos))
        number_halos_inverse[np.abs(number_halos_inverse) == np.inf] = 0
    
    return number_halos_inverse


fig: Figure = plt.figure()
ax: Axes = fig.gca()

num_bins = 30
sim_volume = 100 ** 3

linestyles = ["solid", "dashed", "dotted"]
colors = ["C1", "C2"]

for i, waveform in enumerate(["DB2", "shannon"]):
    for j, resolution in enumerate([128, 256, 512]):
        print(waveform, resolution)
        dir = base_dir / f"{waveform}_{resolution}_100"
        halos = read_velo_halos(dir)

        halos = halos[halos["Mvir"] > 2]  # there seem to be multiple halos with a mass of 1.88196993

        # halos.to_csv("weird_halos.csv")

        halo_masses: np.ndarray = halos["Mvir"].to_numpy()
        bins = np.geomspace(halo_masses.min(), halo_masses.max(), num_bins + 1)

        digits = np.digitize(halo_masses, bins)

        number_densities = []
        widths = []
        centers = []
        left_edges = []
        Ns=[]

        for bin_id in range(num_bins):
            mass_low = bins[bin_id]
            mass_high = bins[bin_id + 1]
            counter = 0
            for val in halo_masses:
                if mass_low <= val < mass_high:
                    counter += 1
            delta_mass = mass_high - mass_low
            widths.append(delta_mass)
            centers.append(mass_low + delta_mass / 2)
            left_edges.append(mass_low)

            values = np.where(digits == bin_id + 1)[0]
            # print(halo_masses[values])
            # print(values)
            num_halos = values.shape[0]
            assert num_halos == counter
            nd = num_halos / sim_volume / delta_mass
            number_densities.append(nd)
            Ns.append(num_halos)

        ax.set_xscale("log")
        ax.set_yscale("log")

        # ax.bar(centers, number_densities, width=widths, log=True, fill=False)
        name = f"{waveform} {resolution}"
        number_densities = np.array(number_densities)
        Ns = np.array(Ns)
        ax.step(left_edges, number_densities, where="post", color=colors[i], linestyle=linestyles[j], label=name)

        lower_error_limit = number_densities - counts_without_inf(Ns) / sim_volume / delta_mass
        upper_error_limit = number_densities + counts_without_inf(Ns) / sim_volume / delta_mass

        ax.fill_between(
            left_edges,
            lower_error_limit,
            upper_error_limit, alpha=.5, linewidth=0, step='post')

        break
    break
plt.legend()
plt.show()
