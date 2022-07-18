from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

# density like in Vr:
from halo_vis import get_comp_id
from paths import base_dir
from utils import figsize_from_page_fraction, rowcolumn_labels

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
            colour = 'white'
        else:
            cnfw = r_200crit / rmax
            colour = 'white'
    else:
        if npart >= 100:  # only calculate cnfw for groups with more than 100 particles
            cnfw = row[f'{halo_type}_cNFW']
            colour = 'black'
        else:
            if m_200crit == 0:
                cnfw = r_size / rmax
                colour = 'white'
            else:
                cnfw = r_200crit / rmax
                colour = 'white'
    assert np.isclose(cnfw, row[f'{halo_type}_cNFW'])

    return cnfw, colour


def plot_comparison_hist2d(ax: Axes, file: Path, property: str, mode: str):
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
    bins = np.geomspace(min_x, max_x, 100)
    if mode == "concentration_bla" and property == 'cNFW':
        colors = []
        for i, row in df.iterrows():
            comp_cnfw, comp_colour = concentration(row, halo_type="comp")  # ref or comp
            ref_cnfw, ref_colour = concentration(row, halo_type='ref')
            if comp_colour == 'white' or ref_colour == 'white':
                colors.append('white')
            else:
                colors.append('black')
        ax.scatter(df[x_col], df[y_col], c=colors, s=1, alpha=.3)
    else:
        rep_row = 20
        rep_x_left = bins[rep_row]
        rep_x_right = bins[rep_row] + 1
        rep_bin = (rep_x_left < df[x_col]) & (df[x_col] < rep_x_right)
        rep_values = df.loc[rep_bin][y_col]
        mean = rep_values.mean()
        std = rep_values.std()
        print(rep_values.describe())
        image: QuadMesh
        xedges: np.ndarray  # 1d
        yedges: np.ndarray  # 1d
        hist: np.ndarray  # 2d
        hist, xedges, yedges, image = ax.hist2d(df[x_col], df[y_col], bins=(bins, bins), norm=LogNorm())
        ax.plot([rep_x_left, rep_x_left], [mean - std, mean + std], c="C1")
        ax.annotate(
            text=f"std={std:.2f}", xy=(rep_x_left, mean + std),
            textcoords="axes fraction", xytext=(0.1, 0.9),
            arrowprops={}
        )
        print(mean - std, mean + std)
        # print(hist)
        # print(list(hist[rep_row]))
        # print(rep_x_left)
        # exit()
        print("vmin/vmax", image.norm.vmin, image.norm.vmax)
        # fig.colorbar(hist)

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.loglog([min_x, max_x], [min_x, max_x], linewidth=1, color="C2")
    return x_col, y_col
    # ax.set_title(file.name)
    # fig.savefig(Path(f"~/tmp/comparison_{file.stem}.pdf").expanduser())
    # fig.suptitle


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
    ax.set_aspect("scaled")
    # fig2.savefig(Path(f"~/tmp/distances_{file.stem}.pdf").expanduser())
    fig2.suptitle
    plt.show()


comparisons_dir = base_dir / "comparisons"

# properties = ['group_size', 'Mass_200crit', 'Mass_tot', 'Mvir', 'R_200crit', 'Rvir', 'Vmax', 'cNFW', 'q',
#               's']  # Mass_FOF and cNFW_200crit don't work, rest looks normal except for cNFW
properties = ['Mvir']
# mode = 'concentration_analysis'
mode = 'normal'
waveforms = ["DB2", "DB4", "DB8", "shannon"]

comparisons = [(256, 512), (256, 1024)]  # , (512, 1024)

for property in properties:
    fig: Figure
    fig, axes = plt.subplots(
        len(waveforms), len(comparisons),
        sharey="all", sharex="all",
        figsize=figsize_from_page_fraction(columns=2)
    )
    for i, waveform in enumerate(waveforms):
        for j, (ref_res, comp_res) in enumerate(comparisons):
            file_id = get_comp_id(waveform, ref_res, waveform, comp_res)
            file = comparisons_dir / file_id
            print(file)
            ax: Axes = axes[i, j]
            x_col, y_col = plot_comparison_hist2d(ax, file, property, mode)
            if i == len(waveforms) - 1:
                ax.set_xlabel(x_col)
            if j == 0:
                ax.set_ylabel(y_col)
    pad = 5
    rowcolumn_labels(axes, comparisons, isrow=False)
    rowcolumn_labels(axes, waveforms, isrow=True)

    fig.tight_layout()
    fig.savefig(Path(f"~/tmp/comparison_{property}.pdf").expanduser())
    plt.show()
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
