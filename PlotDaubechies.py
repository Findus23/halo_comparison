"""
originally created by Oliver Hahn
in PlotDaubechies.ipynb
"""
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# two-fold upsampling -- https://cnx.org/contents/xsppCgXj@8.18:H_wA16rf@16/Upsampling
from matplotlib.figure import Figure
from pyvista import Axes

# # The Cascade Algorithm to Compute the Wavelet and the Scaling Function
# Algorithm adapted from [this webpage](https://cnx.org/contents/0nnvPGYf@7/Computing-the-Scaling-Function-The-Cascade-Algorithm). The iterations are defined by
# $$ \varphi^{(k+1)}(t)=\sqrt{2} \;\sum_{n=0}^{N-1} h[n]\, \varphi^{(k)} (2t-n) $$
# For the $k$th iteration, where an initial $\varphi^{(0)}(t)$ must be given.
#
# The idea of the cascade algorithm is to re-write this as an ordinary convolution, yielding
# $$ \varphi^{(k+1)}_{1/2}[t] := \varphi^{(k+1)} \left( \frac{t}{2} \right) = \sqrt{2} \; \sum_{n=0}^{N-1} h[n]\, \varphi^{(k)} (t-n) $$
# Consider the next iteration (and let $p:=2n$)
# $$
# \begin{align}
# \varphi^{(k+2)} \left( \frac{t}{4} \right) &= \sqrt{2} \; \sum_{n=0}^{N-1} h[n]\, \varphi^{(k+1)} \left(\frac{t}{2}-n\right) \\
# & = \sqrt{2} \; \sum_{p \;{\rm even}} h\left[\frac{p}{2}\right]\, \varphi^{(k+1)} \left(\frac{t-p}{2}\right) \\
# & = \sqrt{2} \; \sum_{p=0}^{N-1} h_{\uparrow 2}\left[p\right]\, \varphi^{(k+1)}_{1/2} \left(t-p\right) \\
# \end{align}
# $$
# which defines the iterations of the cascade algorithm as simple convolutions $\varphi \to \sqrt{2} \,\varphi \ast h$ followed by upsampling of $h \to h_{\uparrow 2}$.
#
# And upsampling is defined in the usual way as ([see also here](https://cnx.org/contents/xsppCgXj@8.18:H_wA16rf@16/Upsampling))
# $$
# x_{\uparrow L}[n] :=
# \left\{ \begin{array}{ll}
# x[n/L] & \textrm{if } \frac{n}{L} \in \mathbb{Z} \\
# 0 & \textrm{otherwise}
# \end{array}\right. .
# $$
#
# Finally, to obtain the wavelet function, we also need the wavelet-scaling equation
# $$
# \psi(t) = \sqrt{2} \sum_{n=0}^{N-1} g[n] \, \varphi(2t-n)
# $$
# which can be applied in the cascade algorithm in the last iteration to obtain the wavelet rather than the scaling function.
#
from utils import figsize_from_page_fraction


def upsample(sig):
    signew = np.zeros(len(sig) * 2 - 1)
    signew[::2] = sig
    return signew


# use the cascade algorithm to compute the scaling function and the wavelet
# -- https://cnx.org/contents/0nnvPGYf@7/Computing-the-Scaling-Function-The-Cascade-Algorithm
def cascade_algorithm(h, g, maxit):
    x = np.arange(len(h), dtype=float)

    phi = np.ones(len(h))
    psi = np.ones(len(h))

    phi_it = np.copy(phi)
    psi_it = np.copy(psi)
    h_it = np.copy(h)
    g_it = np.copy(g)

    for it in range(maxit):
        # perform repeated convolutions
        phi_it = np.sqrt(2) * np.convolve(h_it, phi_it, mode='full')

        if it != maxit - 1:
            psi_it = np.sqrt(2) * np.convolve(h_it, psi_it, mode='full')
        else:
            psi_it = np.sqrt(2) * np.convolve(g_it, psi_it, mode='full')

        # upsample the coefficients
        h_it = upsample(h_it)
        g_it = upsample(g_it)

        # get the new mid-point positions
        xnew = np.zeros(len(x) + len(x) - 1)
        xnew[::2] = x
        xnew[1::2] = x[:-1] + 2 ** -(it + 1)
        x = xnew

    return x, phi_it / len(h), psi_it / len(h)


# # Wavelet Coefficients for DB2 to DB8


maxit = 10

# Haar wavelet
h_Haar = np.array([0.7071067811865476, 0.7071067811865476])
g_Haar = np.array([0.7071067811865476, -0.7071067811865476])

xhaar, phihaar, psihaar = cascade_algorithm(h_Haar, g_Haar, maxit)

# DB2 -- http://wavelets.pybytes.com/wavelet/db2/
h_DB2 = np.array([0.4829629131, 0.8365163037, 0.2241438680, -0.1294095226])
g_DB2 = np.array([-0.1294095226, -0.2241438680, 0.8365163037, -0.4829629131])

xdb2, phidb2, psidb2 = cascade_algorithm(h_DB2, g_DB2, maxit)

# DB3 -- http://wavelets.pybytes.com/wavelet/db3/
h_DB3 = np.array(
    [0.3326705529509569, 0.8068915093133388, 0.4598775021193313, -0.13501102001039084, -0.08544127388224149,
     0.035226291882100656])
g_DB3 = np.array(
    [0.035226291882100656, 0.08544127388224149, -0.13501102001039084, -0.4598775021193313, 0.8068915093133388,
     -0.3326705529509569])

xdb3, phidb3, psidb3 = cascade_algorithm(h_DB3, g_DB3, maxit)

# DB4 -- http://wavelets.pybytes.com/wavelet/db4/
h_DB4 = np.array(
    [0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385, -0.18703481171888114,
     0.030841381835986965, 0.032883011666982945, -0.010597401784997278])
g_DB4 = np.array(
    [-0.010597401784997278, -0.032883011666982945, 0.030841381835986965, 0.18703481171888114, -0.02798376941698385,
     -0.6308807679295904, 0.7148465705525415, -0.23037781330885523])

xdb4, phidb4, psidb4 = cascade_algorithm(h_DB4, g_DB4, maxit)

# DB8 -- http://wavelets.pybytes.com/wavelet/db8/
h_DB8 = np.array(
    [0.05441584224308161, 0.3128715909144659, 0.6756307362980128, 0.5853546836548691, -0.015829105256023893,
     -0.2840155429624281, 0.00047248457399797254, 0.128747426620186, -0.01736930100202211, -0.04408825393106472,
     0.013981027917015516, 0.008746094047015655, -0.00487035299301066, -0.0003917403729959771, 0.0006754494059985568,
     -0.00011747678400228192])
g_DB8 = np.array(
    [-0.00011747678400228192, -0.0006754494059985568, -0.0003917403729959771, 0.00487035299301066, 0.008746094047015655,
     -0.013981027917015516, -0.04408825393106472, 0.01736930100202211, 0.128747426620186, -0.00047248457399797254,
     -0.2840155429624281, 0.015829105256023893, 0.5853546836548691, -0.6756307362980128, 0.3128715909144659,
     -0.05441584224308161])

xdb8, phidb8, psidb8 = cascade_algorithm(h_DB8, g_DB8, maxit)

# DB16 -- 
h_DB16 = np.array(
    [0.0031892209253436892, 0.03490771432362905, 0.1650642834886438, 0.43031272284545874, 0.6373563320829833,
     0.44029025688580486, -0.08975108940236352, -0.3270633105274758, -0.02791820813292813, 0.21119069394696974,
     0.027340263752899923, -0.13238830556335474, -0.006239722752156254, 0.07592423604445779, -0.007588974368642594,
     -0.036888397691556774, 0.010297659641009963, 0.013993768859843242, -0.006990014563390751, -0.0036442796214883506,
     0.00312802338120381, 0.00040789698084934395, -0.0009410217493585433, 0.00011424152003843815,
     0.00017478724522506327, -6.103596621404321e-05, -1.394566898819319e-05, 1.133660866126152e-05,
     -1.0435713423102517e-06, -7.363656785441815e-07, 2.3087840868545578e-07, -2.1093396300980412e-08])
g_DB16 = np.array([-2.1093396300980412e-08, -2.3087840868545578e-07, -7.363656785441815e-07, 1.0435713423102517e-06,
                   1.133660866126152e-05, 1.394566898819319e-05, -6.103596621404321e-05, -0.00017478724522506327,
                   0.00011424152003843815, 0.0009410217493585433, 0.00040789698084934395, -0.00312802338120381,
                   -0.0036442796214883506, 0.006990014563390751, 0.013993768859843242, -0.010297659641009963,
                   -0.036888397691556774, 0.007588974368642594, 0.07592423604445779, 0.006239722752156254,
                   -0.13238830556335474, -0.027340263752899923, 0.21119069394696974, 0.02791820813292813,
                   -0.3270633105274758, 0.08975108940236352, 0.44029025688580486, -0.6373563320829833,
                   0.43031272284545874, -0.1650642834886438, 0.03490771432362905, -0.0031892209253436892])

xdb16, phidb16, psidb16 = cascade_algorithm(h_DB16, g_DB16, maxit)

###################################

fig: Figure
fig, ax = plt.subplots(
    4, 2,
    figsize=figsize_from_page_fraction(height_to_width=12 / 8),
    # sharex="all", sharey="all"
)
labels = ['Haar', 'DB2', 'DB4', 'DB8', 'DB16']

ax[0, 0].set_title('scaling functions $\\varphi$')
ax[0, 1].set_title('wavelets $\\psi$')

ax[0, 0].plot(xhaar, phihaar, lw=1)
ax[0, 1].plot(xhaar, psihaar, lw=1)

ax[1, 0].plot(xdb2, phidb2, lw=1)
ax[1, 1].plot(xdb2, psidb2, lw=1)

ax[2, 0].plot(xdb4, phidb4, lw=1)
ax[2, 1].plot(xdb4, psidb4, lw=1)

ax[3, 0].plot(xdb8, phidb8, lw=1)
ax[3, 1].plot(xdb8, psidb8, lw=1)

# ax[4, 0].plot(xdb16, phidb16, lw=1)
# ax[4, 1].plot(xdb16, psidb16, lw=1)
for a in ax.flatten():
    a.set_xlabel('t')


def inset_label(ax: Axes, text: str):
    ax.text(
        0.75,
        0.05,
        text,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes
    )


for a, label in zip(ax[:, 0], labels):
    text = r"$\varphi_{\textrm{LABEL}}$".replace("LABEL", label)
    inset_label(a, text)
    a.set_ylim([-1.0, 1.5])

for a, label in zip(ax[:, 1], labels):
    text = r"$\psi_{\textrm{LABEL}}$".replace("LABEL", label)
    inset_label(a, text)
    # a.set_ylabel('$\\psi_{\\rm ' + i + '}[t]$')
    a.set_ylim([-2, 2])

fig.tight_layout()
fig.savefig(Path(f"~/tmp/wavelets.pdf").expanduser())

# # Spectral Response of Scaling Functions and Wavelets


fig2: Figure = plt.figure(figsize=figsize_from_page_fraction())
ax: Axes = fig2.gca()


def fourier_wavelet(h, g, n):
    k = np.linspace(0, np.pi, n)
    fphi = np.zeros(n, dtype=complex)
    fpsi = np.zeros(n, dtype=complex)
    N = len(h)
    for i in range(N):
        fphi += np.exp(1j * k * i) * h[i] / np.sqrt(2)
        fpsi += np.exp(1j * k * i) * g[i] / np.sqrt(2)
    return k, fphi, fpsi


# ax.plot([0, np.pi], [0., 0.], 'k:')
# ax.plot([0, np.pi], [1., 1.], 'k:')

kh, fphih, fpsih = fourier_wavelet(h_Haar, g_Haar, 256)
ax.plot(kh, np.abs(fphih) ** 2, label='$\\hat\\varphi_{\\rm Haar}$', c="C0")
ax.plot(kh, np.abs(fpsih) ** 2, label='$\\hat\\psi_{\\rm Haar}$', c="C0", linestyle="dashed")

kdb2, fphidb2, fpsidb2 = fourier_wavelet(h_DB2, g_DB2, 256)
ax.plot(kdb2, np.abs(fphidb2) ** 2, label='$\\hat\\varphi_{DB2}$', c="C1")
ax.plot(kdb2, np.abs(fpsidb2) ** 2, label='$\\hat\\psi_{DB2}$', c="C1", linestyle="dashed")

kdb4, fphidb4, fpsidb4 = fourier_wavelet(h_DB4, g_DB4, 256)
ax.plot(kdb4, np.abs(fphidb4) ** 2, label='$\\hat\\varphi_{DB4}$', c="C2")
ax.plot(kdb4, np.abs(fpsidb4) ** 2, label='$\\hat\\psi_{DB4}$', c="C2", linestyle="dashed")

kdb8, fphidb8, fpsidb8 = fourier_wavelet(h_DB8, g_DB8, 256)
ax.plot(kdb8, np.abs(fphidb8) ** 2, label='$\\hat\\varphi_{DB8}$', c="C3")
ax.plot(kdb8, np.abs(fpsidb8) ** 2, label='$\\hat\\psi_{DB8}$', c="C3", linestyle="dashed")


# all k* are np.linspace(0, np.pi, 256), so we can also use them for shannon

def shannon(k):
    y = np.zeros_like(k)
    y[k > pi / 2] = 1
    return y


ax.plot(kdb8, 1 - shannon(kdb8), label='$\\hat\\varphi_{shannon}$', c="C4")
ax.plot(kdb8, shannon(kdb8), label='$\\hat\\psi_{shannon}$', c="C4", linestyle="dashed")
# ax.plot(kdb8, np.abs(fpsidb8) ** 2, label='$\\hat\\psi_{DB8}$', c="C3", linestyle="dashed")

# kdb16, fphidb16, fpsidb16 = fourier_wavelet(h_DB16, g_DB16, 256)
# ax.plot(kdb16, np.abs(fphidb16) ** 2, label='$\\hat\\varphi_{DB16}$', c="C4")
# ax.plot(kdb16, np.abs(fpsidb16) ** 2, label='$\\hat\\psi_{DB16}$', c="C4", linestyle="dashed")
ax.legend(frameon=False)
ax.set_xlabel('k')
ax.set_ylabel('P(k)')
ax.set_xticks([0, pi / 2, pi])
ax.set_xticklabels(["0", r"$k_\textrm{coarse}^\textrm{ny}$", r"$k_\textrm{fine}^\textrm{ny}$"])

# plt.semilogy()
# plt.ylim([1e-4,2.0])
fig2.tight_layout()
fig2.savefig(Path(f"~/tmp/wavelet_power.pdf").expanduser())
plt.show()
