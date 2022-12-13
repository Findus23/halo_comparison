import matplotlib.pyplot as plt
from velociraptor import load, tools

from paths import base_dir
from read_vr_files import read_velo_halos
from unyt import Mpc
import unyt

for i, waveform in enumerate(["DB2", "DB4", "DB8", "shannon"]):
    for j, resolution in enumerate([128, 256, 512]):
        print(waveform, resolution)
        dir = base_dir / f"{waveform}_{resolution}_100"
        data = load(str(dir)+"/vroutput.properties.0")
        print(data)
        masses_200crit = data.masses.mass_200crit
        masses_200crit.convert_to_units("msun")
        lowest_halo_mass = 1e9 * unyt.msun
        highest_halo_mass = 8e15 * unyt.msun
        bin_centers, mass_function, error = tools.create_mass_function(
            masses_200crit, lowest_halo_mass, highest_halo_mass, (100*Mpc)**3
        )
        plt.loglog(bin_centers,mass_function)
        # halos = read_velo_halos(dir)
        #
        # print(halos["Rvir"].sort_values(ascending=False)[:4].to_numpy())


plt.show()
