from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from halo_vis import get_comp_id
from paths import base_dir
from utils import figsize_from_page_fraction, rowcolumn_labels, waveforms, tex_fmt

# density like in Vr:

G = 43.022682  # in Mpc (km/s)^2 / (10^10 Msun)

vmaxs = {
    "Mvir": 52,
    "Vmax": 93,
    "cNFW": 31
}

units = {
    "distance": "Mpc",
    "Mvir": r"10^{10} M_\odot",
    "Vmax": "???"  # TODO
}


def concentration(row, halo_type: str) -> bool:
    r_200crit = row[f'{halo_type}_R_200crit']
    if r_200crit <= 0:
        cnfw = -1
        colour = 'orange'
        return False
        # return cnfw, colour

    r_size = row[f'{halo_type}_R_size']  # largest difference from center of mass to any halo particle
    m_200crit = row[f'{halo_type}_Mass_200crit']
    vmax = row[f'{halo_type}_Vmax']  # largest velocity coming from enclosed mass profile calculation
    rmax = row[f'{halo_type}_Rmax']
    npart = row[f'{halo_type}_npart']
    VmaxVvir2 = vmax ** 2 * r_200crit / (G * m_200crit)
    if VmaxVvir2 <= 1.05:
        if m_200crit == 0:
            cnfw = r_size / rmax
            return False
            # colour = 'white'
        else:
            cnfw = r_200crit / rmax
            return False
            # colour = 'white'
    else:
        if npart >= 100:  # only calculate cnfw for groups with more than 100 particles
            cnfw = row[f'{halo_type}_cNFW']
            return True
            # colour = 'black'
        else:
            if m_200crit == 0:
                cnfw = r_size / rmax
                return False
                # colour = 'white'
            else:
                cnfw = r_200crit / rmax
                return False
                # colour = 'white'
    # assert np.isclose(cnfw, row[f'{halo_type}_cNFW'])
    #
    # return cnfw, colour


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
    num_bins = 100
    bins = np.geomspace(min_x, max_x, num_bins)
    if property == 'cNFW':
        rows = []
        for i, row in df.iterrows():
            comp_cnfw_normal = concentration(row, halo_type="comp")

            ref_cnfw_normal = concentration(row, halo_type='ref')
            cnfw_normal = comp_cnfw_normal and ref_cnfw_normal
            if cnfw_normal:
                rows.append(row)
        df = pd.concat(rows, axis=1).T
        print(df)
    if property == "Mvir":
        stds = []
        means = []
        for rep_row in range(num_bins):
            rep_x_left = bins[rep_row]
            rep_x_right = bins[rep_row] + 1
            rep_bin = (rep_x_left < df[x_col]) & (df[x_col] < rep_x_right)
            rep_values = df.loc[rep_bin][y_col] / df.loc[rep_bin][x_col]
            if len(rep_bin) < 30:
                continue
            mean = rep_values.mean()
            std = rep_values.std()
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        args = {
            "color": "C2",
            "zorder": 10
        }
        ax.fill_between(bins, means - stds, means + stds, alpha=.2, **args)
        ax.plot(bins, means + stds, alpha=.5, **args)
        ax.plot(bins, means - stds, alpha=.5, **args)
        # ax_scatter.plot(bins, stds, label=f"{file.stem}")

    if property in vmaxs:
        vmax = vmaxs[property]
    else:
        vmax = None
        print("WARNING: vmax not set")
    image: QuadMesh
    _, _, _, image = ax.hist2d(df[x_col], df[y_col] / df[x_col], bins=(bins, np.linspace(0, 2, num_bins)),
                               norm=LogNorm(vmax=vmax))
    # ax.plot([rep_x_left, rep_x_left], [mean - std, mean + std], c="C1")
    # ax.annotate(
    #     text=f"std={std:.2f}", xy=(rep_x_left, mean + std),
    #     textcoords="axes fraction", xytext=(0.1, 0.9),
    #     arrowprops={}
    # )
    print("vmin/vmax", image.norm.vmin, image.norm.vmax)
    # fig.colorbar(hist)

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlim(min(df[x_col]), max(df[y_col]))

    ax.plot([min(df[x_col]), max(df[y_col])], [1, 1], linewidth=1, color="C1", zorder=10)

    return x_col, y_col
    # ax.set_title(file.name)
    # fig.savefig(Path(f"~/tmp/comparison_{file.stem}.pdf").expanduser())
    # fig.suptitle


def plot_comparison_hist(ax: Axes, file: Path, property: str, mode: str):
    df = pd.read_csv(file)
    if mode == 'concentration_analysis':
        df = df.loc[2 * df.ref_cNFW < df.comp_cNFW]

    ax.hist(df[property][df[property] < 50], bins=100)


comparisons_dir = base_dir / "comparisons"
hist_properties = ["distance", "match", "num_skipped_for_mass"]

comparisons = [(256, 512), (256, 1024)]  # , (512, 1024)


def compare_property(property, mode, show: bool):
    is_hist_property = property in hist_properties
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
            is_bottom_row = i == len(waveforms) - 1
            is_left_col = j == 0
            if not is_hist_property:
                x_labels = {
                    "Mvir": ("M", "vir"),
                    "Vmax": ("V", "max"),
                    "cNFW": ("c", None),
                }
                x_col, y_col = plot_comparison_hist2d(ax, file, property, mode)
                lab_a, lab_b = x_labels[property]
                unit = f"[{units[property]}]" if property in units and units[property] else ""
                if is_bottom_row:
                    if lab_b:
                        ax.set_xlabel(tex_fmt(r"$AA_{\textrm{BB},CC} DD$", lab_a, lab_b, ref_res, unit))
                    else:
                        ax.set_xlabel(tex_fmt(r"$AA_{BB} CC$", lab_a, ref_res, unit))
                if is_left_col:
                    if lab_b:
                        ax.set_ylabel(
                            tex_fmt(r"$AA_{\textrm{BB},\textrm{comp}} / AA_{\textrm{BB},\textrm{CC}}$",
                                    lab_a, lab_b, ref_res))
                    else:
                        ax.set_ylabel(
                            tex_fmt(r"$AA_{\textrm{comp}} / AA_{\textrm{BB}}$",
                                    lab_a, ref_res))
                    # ax.set_ylabel(f"{property}_{{comp}}/{property}_{ref_res}")
            else:
                plot_comparison_hist(ax, file, property, mode)
                if is_bottom_row:
                    x_labels = {
                        "match": "$J$",
                        "distance": "$R$"
                    }
                ax.set_xlabel(x_labels[property])
                if is_left_col:
                    ax.set_ylabel(r"\# Halos")

    rowcolumn_labels(axes, comparisons, isrow=False)
    rowcolumn_labels(axes, waveforms, isrow=True)
    fig.tight_layout()
    fig.savefig(Path(f"~/tmp/comparison_{property}.pdf").expanduser())
    if show:
        plt.show()


def main():
    # properties = ['group_size', 'Mass_200crit', 'Mass_tot', 'Mvir', 'R_200crit', 'Rvir', 'Vmax', 'cNFW', 'q',
    #               's']  # Mass_FOF and cNFW_200crit don't work, rest looks normal except for cNFW
    if len(argv) > 1:
        properties = argv[1:]
    else:
        properties = ["Mvir", "Vmax", "cNFW"]
    # mode = 'concentration_analysis'

    mode = 'concentration_bla'

    for property in properties:
        compare_property(property, mode, show=len(argv) == 2)


if __name__ == '__main__':
    main()
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
