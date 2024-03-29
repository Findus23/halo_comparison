from pathlib import Path
from sys import argv
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import YTick
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from numpy import inf
from scipy.stats import norm

from halo_vis import get_comp_id
from paths import base_dir
from utils import figsize_from_page_fraction, rowcolumn_labels, waveforms, tex_fmt

# density like in Vr:

G = 43.022682  # in Mpc (km/s)^2 / (10^10 Msun)

vmaxs = {"Mvir": 52, "Vmax": 93, "cNFW": 31}

units = {
    "distance": "Mpc",
    "Mvir": r"10^{10} \textrm{ M}_\odot",
    "Vmax": r"\textrm{km } \textrm{s}^{-1}",  # TODO
}


def concentration(row, halo_type: str) -> bool:
    r_200crit = row[f"{halo_type}_R_200crit"]
    if r_200crit <= 0:
        cnfw = -1
        colour = "orange"
        return False
        # return cnfw, colour

    r_size = row[
        f"{halo_type}_R_size"
    ]  # largest difference from center of mass to any halo particle
    m_200crit = row[f"{halo_type}_Mass_200crit"]
    vmax = row[
        f"{halo_type}_Vmax"
    ]  # largest velocity coming from enclosed mass profile calculation
    rmax = row[f"{halo_type}_Rmax"]
    npart = row[f"{halo_type}_npart"]
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
            cnfw = row[f"{halo_type}_cNFW"]
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


def plot_comparison_hist2d(ax: Axes, file: Path, property: str):
    print("WARNING: Can only plot hist2d of properties with comp_ or ref_ right now!")
    print(f"         Selected property: {property}")
    x_col = f"ref_{property}"
    y_col = f"comp_{property}"
    df = pd.read_csv(file)
    # if mode == 'concentration_analysis':
    #     min_x = min([min(df[x_col]), min(df[y_col])])
    #     max_x = max([max(df[x_col]), max(df[y_col])])
    #     df = df.loc[2 * df.ref_cNFW < df.comp_cNFW]
    # else:
    min_x = min([min(df[x_col]), min(df[y_col])])
    max_x = max([max(df[x_col]), max(df[y_col])])
    num_bins = 100
    bins = np.geomspace(min_x, max_x, num_bins)
    if property == "cNFW":
        rows = []
        for i, row in df.iterrows():
            comp_cnfw_normal = concentration(row, halo_type="comp")

            ref_cnfw_normal = concentration(row, halo_type="ref")
            cnfw_normal = comp_cnfw_normal and ref_cnfw_normal
            if cnfw_normal:
                rows.append(row)
        df = pd.concat(rows, axis=1).T
        print(df)
    if property in ["Mvir", "Vmax"]:
        percentiles = []
        for rep_row in range(num_bins):
            rep_x_left = bins[rep_row]
            rep_x_right = bins[rep_row] + 1
            rep_bin = (rep_x_left < df[x_col]) & (df[x_col] < rep_x_right)
            rep_values = df.loc[rep_bin][y_col] / df.loc[rep_bin][x_col]
            if len(rep_values) < 10:
                percentiles.append([np.nan, np.nan, np.nan])
                continue
            percentiles.append(np.quantile(rep_values, [norm.cdf(-1), norm.cdf(0), norm.cdf(1)]))

        percentiles = np.asarray(percentiles)
        print(percentiles.shape)
        args = {"color": "C1", "zorder": 10}
        # ax.fill_between(bins, percentiles[::, 0], percentiles[::, 2], alpha=0.1, **args)
        ax.plot(bins, percentiles[::, 0], alpha=0.9, **args)
        ax.plot(bins, percentiles[::, 1], alpha=0.9, **args)
        ax.plot(bins, percentiles[::, 2], alpha=0.9, **args)

    if property in vmaxs:
        vmax = vmaxs[property]
    else:
        vmax = None
        print("WARNING: vmax not set")
    image: QuadMesh
    _, _, _, image = ax.hist2d(
        df[x_col],
        df[y_col] / df[x_col],
        bins=(bins, np.linspace(0, 2, num_bins)),
        norm=LogNorm(vmax=vmax),
        cmap="gray_r",
        rasterized=True,
    )
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

    ax.plot(
        [min(df[x_col]), max(df[y_col])], [1, 1], linewidth=1, color="C0", zorder=10, linestyle="dotted"
    )

    return x_col, y_col
    # ax.set_title(file.name)
    # fig.savefig(Path(f"~/tmp/comparison_{file.stem}.pdf").expanduser())
    # fig.suptitle


def plot_comparison_hist(ax: Axes, file: Path, property: str, m_min=None, m_max=None):
    df = pd.read_csv(file)
    if m_min:
        df = df.loc[(m_min < df["ref_Mvir"]) & (df["ref_Mvir"] < m_max)]

    num_bins = 100
    histtype = "bar"
    label = None
    density = False

    if property == "distance":
        bins = np.geomspace(min(df[property]), max(df[property]), 100)
        mean = df[property].mean()
        median = df[property].median()
        ax.axvline(mean, label="mean", color="C1")
        ax.axvline(median, label="median", color="C2")
    else:
        bins = num_bins
    if property == "match":
        labels = {
            (-inf, 30): "$M<30$",
            (None, None): "all $M$ $[10^{10}$ $\mathrm{M}_\odot]$",
            (30, 100): "$30<M<100$",
            (100, inf): "$100<M$",
        }
        colours = {
            (-inf, 30): 'C1',
            (None, None): 'C0', 
            (30, 100): 'C2',
            (100, inf): 'C3',
        }
        label = labels[(m_min, m_max)]
        colour = colours[(m_min, m_max)]
        density = True
    if property == "match":
        hist_val, bin_edges = np.histogram(df[property], bins=bins, density=density)
        bin_centers = []
        for i in range(len(hist_val)):
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        ax.plot(bin_centers, hist_val, label=label)
        median = np.median(df[property])
        ax.axvline(median, linestyle=":", color=colour)
    else:
        patches: List[Polygon]
        hist_val, bin_edges, patches = ax.hist(
            df[property], bins=bins, histtype="step", label=label, density=density
        )


comparisons_dir = base_dir / "comparisons"
hist_properties = ["distance", "match", "num_skipped_for_mass"]

comparisons = [(256, 512), (256, 1024)]  # , (512, 1024)


def compare_property(property, show: bool):
    is_hist_property = property in hist_properties
    height_to_width = 2 / 4
    if property == "distance":
        height_to_width = 3/8
    fig: Figure
    fig, axes = plt.subplots(
        len(waveforms),
        len(comparisons),
        sharey="all",
        sharex="all",
        figsize=figsize_from_page_fraction(columns=2, height_to_width=height_to_width),
    )
    for i, waveform in enumerate(waveforms):
        for j, (ref_res, comp_res) in enumerate(comparisons):
            file_id = get_comp_id(waveform, ref_res, waveform, comp_res)
            file = comparisons_dir / file_id
            print(file)
            ax: Axes = axes[i, j]
            is_bottom_row = i == len(waveforms) - 1
            is_top_row = i == 0
            is_left_col = j == 0
            if not is_hist_property:
                x_labels = {
                    "Mvir": ("M", "vir"),
                    "Vmax": ("V", "max"),
                    "cNFW": ("C", None),
                }
                x_col, y_col = plot_comparison_hist2d(ax, file, property)
                lab_a, lab_b = x_labels[property]
                unit = (
                    f"[{units[property]}]"
                    if property in units and units[property]
                    else ""
                )
                if is_bottom_row:
                    if lab_b:
                        ax.set_xlabel(
                            tex_fmt(
                                r"$AA_{\textrm{BB},\textrm{ CC}} \textrm{ } DD$",
                                lab_a,
                                lab_b,
                                ref_res,
                                unit,
                            )
                        )
                        # fig.supxlabel(tex_fmt(r"$AA_{\textrm{BB},\textrm{ } CC} \textrm{ } DD$", lab_a, lab_b, ref_res, unit), fontsize='medium')
                    else:
                        ax.set_xlabel(
                            tex_fmt(
                                r"$AA_{\textrm{BB}} \textrm{ } CC$",
                                lab_a,
                                ref_res,
                                unit,
                            )
                        )
                        # fig.supxlabel(tex_fmt(r"$AA_{BB} \textrm{ } CC$", lab_a, ref_res, unit), fontsize='medium')
                if is_left_col:
                    if lab_b:
                        # ax.set_ylabel(
                        #     tex_fmt(r"$AA_{\textrm{BB},\textrm{comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB},\textrm{CC}}$",
                        #             lab_a, lab_b, ref_res))
                        # fig.text(0.015, 0.5, tex_fmt(r"$AA_{\textrm{BB},\textrm{ comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB},\textrm{ CC}}$", lab_a, lab_b, ref_res), va='center', rotation='vertical', size='medium')
                        fig.supylabel(
                            tex_fmt(
                                r"$AA_{\textrm{BB},\textrm{ comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB},\textrm{ CC}}$",
                                lab_a,
                                lab_b,
                                ref_res,
                            ),
                            fontsize="medium",
                            fontvariant="small-caps",
                        )
                    else:
                        # ax.set_ylabel(
                        #     tex_fmt(r"$AA_{\textrm{comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB}}$",
                        #             lab_a, ref_res))
                        # fig.text(0.015, 0.5, tex_fmt(r"$AA_{\textrm{comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB}}$", lab_a, ref_res), va='center', rotation='vertical', size='medium')
                        fig.supylabel(
                            tex_fmt(
                                r"$AA_{\textrm{comp}} \textrm{ } / \textrm{ } AA_{\textrm{BB}}$",
                                lab_a,
                                ref_res,
                            ),
                            fontsize="medium",
                        )
                    # ax.set_ylabel(f"{property}_{{comp}}/{property}_{ref_res}")
                ax.text(
                    0.975,
                    0.9,
                    f"comp = {comp_res}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                )
            else:
                if property == "match":
                    if not (is_bottom_row and is_left_col):
                        ax.text(
                            0.05,
                            0.9,
                            f"comp = {comp_res}",
                            horizontalalignment="left",
                            verticalalignment="top",
                            transform=ax.transAxes,
                        )
                    # mass_bins = np.geomspace(10, 30000, num_mass_bins)
                    plot_comparison_hist(ax, file, property)

                    mass_bins = [-inf, 30, 100, inf]
                    for k in range(len(mass_bins) - 1):
                        m_min = mass_bins[k]
                        m_max = mass_bins[k + 1]
                        plot_comparison_hist(ax, file, property, m_min, m_max)
                    if is_bottom_row and is_left_col:
                        ax.legend()

                else:
                    ax.text(
                        0.05,
                        0.9,
                        f"comp = {comp_res}",
                        horizontalalignment="left",
                        verticalalignment="top",
                        transform=ax.transAxes,
                    )
                    plot_comparison_hist(ax, file, property)
                x_labels = {"match": "$J$", "distance": "$D$ [$R_\mathrm{{vir}}$]"}
                if is_bottom_row:
                    ax.set_xlabel(x_labels[property])
                if is_left_col:
                    if property == "match":
                        # ax.set_ylabel(r"$p(J)$")
                        fig.supylabel(r"$p(J)$", fontsize="medium")
                    else:
                        # ax.set_ylabel(r"\# Halos")
                        fig.supylabel(r"\# Halos", fontsize="medium")
            if property == "distance":
                ax.set_xscale("log")
                ax.set_yscale("log")
                if is_bottom_row and is_left_col:
                    ax.legend()
            if property == "Mvir":
                ax.set_ylim(0.5, 1.5)
            if property == "Vmax":
                ax.set_ylim(0.7, 1.3)
            if not is_top_row:
                last_ytick: YTick = ax.yaxis.get_major_ticks()[-1]
                last_ytick.set_visible(False)
            if property == "Mvir" and is_top_row:
                particle_masses = {256: 0.23524624, 512: 0.02940578, 1024: 0.0036757225}
                partmass = particle_masses[ref_res]

                def mass2partnum(mass: float) -> float:
                    return mass / partmass

                def partnum2mass(partnum: float) -> float:
                    return partnum * partmass

                sec_ax = ax.secondary_xaxis(
                    "top", functions=(mass2partnum, partnum2mass)
                )
                sec_ax.set_xlabel(r"\textrm{Halo Size }[\# \textrm{particles}]")

    # rowcolumn_labels(axes, comparisons, isrow=False)
    rowcolumn_labels(axes, waveforms, isrow=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(Path(f"~/tmp/comparison_{property}.pdf").expanduser())
    if show:
        plt.show()


def main():
    # properties = ['group_size', 'Mass_200crit', 'Mass_tot', 'Mvir', 'R_200crit', 'Rvir', 'Vmax', 'cNFW', 'q',
    #               's']
    if len(argv) > 1:
        properties = argv[1:]
    else:
        properties = ["Mvir", "Vmax", "cNFW", "distance", "match"]

    for property in properties:
        compare_property(property, show=len(argv) == 2)


if __name__ == "__main__":
    main()
