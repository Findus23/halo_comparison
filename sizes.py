from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

# density like in Vr:
G = 43.022682  # in Mpc (km/s)^2 / (10^10 Msun)


def concentration(row, halo_type: str):
    r_200crit = row[f'{halo_type}_R_200crit']
    if r_200crit <= 0:
        cnfw = -1
        colour = 'orange'
        return cnfw, colour

    r_size = row[f'{halo_type}_R_size']  # largest difference from center of mass to any halo particle
    m_200crit = row[f'{halo_type}_Mass_200crit']
    vmax = row[f'{halo_type}_Vmax']  # largest velocity coming from enclosed mass profile calculation
    rmax = row[f'{halo_type}_Rmax']
    npart = row[f'{halo_type}_npart']
    VmaxVvir2 = vmax ** 2 * r_200crit / (G * m_200crit)
    if VmaxVvir2 <= 1.05:
        if m_200crit == 0:
            cnfw = r_size / rmax
            colour = 'red'
        else:
            cnfw = r_200crit / rmax
            colour = 'green'
    else:
        if npart >= 100:  # only calculate cnfw for groups with more than 100 particles
            cnfw = row[f'{halo_type}_cNFW']
            colour = 'black'
        else:
            if m_200crit == 0:
                cnfw = r_size / rmax
                colour = 'blue'
            else:
                cnfw = r_200crit / rmax
                colour = 'purple'
    assert np.isclose(cnfw, row[f'{halo_type}_cNFW'])


    return cnfw, colour


def plot_comparison_hist2d(file: Path, property: str, mode: str):
    print("WARNING: Can only plot hist2d of properties with comp_ or ref_ right now!")
    print(f"         Selected property: {property}")
    x_col = f"ref_{property}"
    y_col = f"comp_{property}"
    df = pd.read_csv(file)
    if mode == 'concentration_analysis':
        min_x = min([min(df[x_col]), min(df[y_col])])
        max_x = max([max(df[x_col]), max(df[y_col])])
        df = df.loc[2 * df.ref_cNFW < df.comp_cNFW]
    else:
        min_x = min([min(df[x_col]), min(df[y_col])])
        max_x = max([max(df[x_col]), max(df[y_col])])
    fig: Figure = plt.figure()
    ax: Axes = fig.gca()
    bins = np.geomspace(min_x, max_x, 100)
    if mode == "concentration_bla" and property == 'cNFW':
        colors = []
        for i, row in df.iterrows():
            cnfw, colour = concentration(row, halo_type="comp") # ref or comp
            colors.append(colour)
        ax.scatter(df[x_col], df[y_col], c=colors, s=1, alpha=.3)
    else:
        _, _, _, hist = ax.hist2d(df[x_col], df[y_col], bins=(bins, bins), norm=LogNorm())
        fig.colorbar(hist)

    # ax.set_xscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    # ax.set_yscale("log")

    ax.loglog([min_x, max_x], [min_x, max_x], linewidth=1, color="C2")
    ax.set_title(file.name)
    # fig.savefig(Path(f"~/tmp/comparison_{file.stem}.pdf").expanduser())
    fig.suptitle
    plt.show()


def plot_comparison_hist(file: Path, property: str, mode: str):
    print("WARNING: Can only plot hist of properties w/o comp_ or ref_ right now!")
    print(f"         Selected property: {property}")
    df = pd.read_csv(file)
    if mode == 'concentration_analysis':
        df = df.loc[2 * df.ref_cNFW < df.comp_cNFW]
    fig2: Figure = plt.figure()
    ax2: Axes = fig2.gca()

    ax2.hist(df[property][df[property] < 50], bins=100)
    ax2.set_xlabel(property)
    ax2.set_title(file.name)
    # fig2.savefig(Path(f"~/tmp/distances_{file.stem}.pdf").expanduser())
    fig2.suptitle
    plt.show()


file = Path(argv[1])

# properties = ['group_size', 'Mass_200crit', 'Mass_tot', 'Mvir', 'R_200crit', 'Rvir', 'Vmax', 'cNFW', 'q', 's'] #Mass_FOF and cNFW_200crit don't work, rest looks normal except for cNFW
properties = ['cNFW']
# mode = 'concentration_analysis'
mode = 'concentration_bla'

for property in properties:
    plot_comparison_hist2d(file, property, mode)

# axis_ratios = ['q', 's'] #they look normal

# for property in axis_ratios:
#     plot_comparison_hist2d(file, property, 'no')
#     plot_comparison_hist2d(file, property, mode)

# plot_comparison_hist2d(file, 'cNFW_200mean', mode)

# ref_property = 'ref_cNFW_200crit'
# comp_property = 'comp_cNFW_200crit'

# df = pd.read_csv(file)
# all_ref_structure_types: pd.DataFrame = df[ref_property]
# all_comp_structure_types: pd.DataFrame = df[comp_property]

# df_odd: pd.DataFrame = df.loc[2 * df.ref_cNFW < df.comp_cNFW]
# odd_ref_structure_types: pd.DataFrame = df_odd[ref_property]
# odd_comp_structure_types: pd.DataFrame = df_odd[comp_property]

# print(all_ref_structure_types.mean(), all_comp_structure_types.mean())
# print(odd_ref_structure_types.mean(), odd_comp_structure_types.mean())


# ref_colour = []
# comp_colour = []
# ref_cnfw = []
# comp_cnfw = []
# df = pd.read_csv(file)
#
# for index, row in df.iterrows():
#     cnfw, colour = concentration(row)
#     ref_cnfw.append(cnfw[0])
#     ref_colour.append(colour[0])
#     comp_cnfw.append(cnfw[1])
#     comp_colour.append(colour[1])
#
# fig: Figure = plt.figure()
# ax: Axes = fig.gca()
#
# ax.scatter(ref_cnfw, comp_cnfw, s=1, c=comp_colour, alpha=.3)
# ax.set_xscale("log")
# ax.set_yscale("log")
# plt.show()

# #Maybe for later:
# if __name__ == '__main__':
#     print('Run with sizes.py <Path to file> <property: str> <mode: str>')
#     file = Path(argv[1])
#     property = str(argv[2])
#     mode = str(argv[3])


# #This is to find the median of the quality of our matches
# matches:pd.DataFrame=df["match"]
# print(matches)
# exit()
# print(matches.median())
# print(matches.std())
# exit()

# #This is to save weird concentration data to own csv
# df_odd: pd.DataFrame = df.loc[2 * df.ref_cNFW < df.comp_cNFW]
# df_odd.to_csv("weird_cnfw.csv")
# exit()
