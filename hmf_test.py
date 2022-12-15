import matplotlib.pyplot as plt

import read_vr_files
from paths import base_dir

for i, waveform in enumerate(
        ["DB2", "DB8", "shannon"]):
    for j, resolution in enumerate([128, 256, 512]):
        if (waveform == "shannon_rehalo" or "resim" in waveform) and resolution != 128:
            continue
        print(waveform, resolution)
        dir = base_dir / f"{waveform}_{resolution}_100"
        # data = load(str(dir) + "/vroutput.properties.0")
        # print(data)
        # masses_200crit = data.masses.mass_200crit
        # masses_200crit.convert_to_units("msun")
        # lowest_halo_mass = 1e9 * unyt.msun
        # highest_halo_mass = 8e15 * unyt.msun
        # bin_centers, mass_function, error = tools.create_mass_function(
        #     masses_200crit, lowest_halo_mass, highest_halo_mass, (100 * Mpc) ** 3
        # )
        #
        # name = f"{waveform} {resolution}"

        # plt.loglog(bin_centers, mass_function, label=name)
        # halos = read_halo_file(dir/"fof_output_0004.hdf5")
        #
        # print(halos["Masses"].sort_values(ascending=False)[:4].to_numpy())

        halos = read_vr_files.read_velo_halos(dir)
        print(halos["R_200mean"].sort_values(ascending=False)[:4].to_numpy())

plt.legend()
plt.show()
