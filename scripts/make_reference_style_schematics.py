"""Reference-style, symbol-led schematics for the CSE-DKAN-FV principle."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path("paper_build/latex/figures/revised_principle_set")

INK = "#171C24"
GREY = "#6A737D"
LIGHT = "#F6F8FA"
BLUE = "#3478B4"
BLUE_LIGHT = "#DCECF8"
RED = "#C94D5B"
RED_LIGHT = "#F7E1E4"
MAGENTA = "#A94E76"
MAGENTA_LIGHT = "#F1DFE8"
GOLD = "#C58A2A"
GOLD_LIGHT = "#F6E9CF"
GREEN = "#4B8D59"
GREEN_LIGHT = "#DFEEDF"
VIOLET = "#7957A5"
VIOLET_LIGHT = "#E9E1F2"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
    }
)


def canvas(size=(12.0, 5.3)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def rounded(ax, x, y, w, h, fc, ec, lw=1.6, r=0.025, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, fs=9, c=INK, w="normal", ha="center", va="center", z=9):
    return ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, zorder=z)


def arr(ax, a, b, c=INK, lw=1.7, ms=13, style="-|>", rad=0, z=6, ls="-"):
    p = FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=ms, color=c, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2, zorder=z,
    )
    ax.add_patch(p)
    return p


def wave(ax, x, y, w, h, kind, c, lw=2.2, z=7):
    xx = np.linspace(0, 1, 180)
    if kind == "low":
        yy = 0.50 + 0.28 * np.tanh((0.52 - xx) * 5)
    elif kind == "high":
        yy = np.where(xx < 0.54, 0.78, 0.22)
    elif kind == "blend":
        yy = 0.50 + 0.29 * np.tanh((0.54 - xx) * 25)
    elif kind == "sine_low":
        yy = 0.50 + 0.22 * np.sin(2 * np.pi * xx)
    elif kind == "sine_high":
        yy = 0.50 + 0.20 * np.sin(8 * np.pi * xx)
    elif kind == "spline":
        yy = 0.48 + 0.23 * np.sin(1.6 * np.pi * xx) + 0.07 * np.sin(4.2 * np.pi * xx)
    elif kind == "tanh":
        yy = 0.50 + 0.30 * np.tanh((xx - 0.52) * 10)
    else:
        raise ValueError(kind)
    ax.plot(x + w * xx, y + h * yy, color=c, lw=lw, solid_capstyle="round", zorder=z)


def lock_icon(ax, x, y, c=GREY, scale=1.0):
    """Compact lock glyph for frozen/non-trainable numerical operators."""
    ax.add_patch(Rectangle((x - 0.010 * scale, y - 0.010 * scale),
                           0.020 * scale, 0.017 * scale,
                           facecolor="white", edgecolor=c, lw=1.0, zorder=9))
    ax.add_patch(Arc((x, y + 0.008 * scale), 0.016 * scale, 0.020 * scale,
                     theta1=0, theta2=180, color=c, lw=1.0, zorder=9))


def tag(ax, x, y, s, fc, ec, w=0.10):
    rounded(ax, x - w / 2, y - 0.018, w, 0.036, fc, ec, 0.9, 0.010, z=7)
    txt(ax, x, y, s, 7.0, ec, "bold", z=10)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1_multiscale_encoder():
    fig, ax = canvas((11.8, 4.9))

    txt(ax, 0.018, 0.955, "a", 11, INK, "bold", ha="left")
    txt(ax, 0.355, 0.905, "EDGE-WISE FUSION", 7.2, GREY, "bold")

    # Three compact physical inputs.
    inputs = [
        (0.12, r"$\bar{\mathbf{U}}_{i-2:i+2}$", BLUE),
        (0.33, r"$\Delta p,\;[-\Delta u]_+$", RED),
        (0.54, r"$r_q$", GOLD),
    ]
    for x, lab, col in inputs:
        ax.add_patch(Circle((x, 0.12), 0.052, facecolor="white", edgecolor=col, lw=1.8, zorder=5))
        txt(ax, x, 0.12, lab, 9.3, col, "bold")

    # High- and low-frequency channels echo the supplied reference but encode the new model.
    rounded(ax, 0.08, 0.31, 0.39, 0.36, RED_LIGHT, RED, 1.5, 0.035)
    rounded(ax, 0.49, 0.31, 0.39, 0.36, BLUE_LIGHT, BLUE, 1.5, 0.035)
    txt(ax, 0.06, 0.50, "JUMP", 10, RED, "bold", ha="right")
    txt(ax, 0.90, 0.50, "SMOOTH", 10, BLUE, "bold", ha="left")
    tag(ax, 0.275, 0.655, "localized / high-$k$", "white", RED, 0.16)
    tag(ax, 0.685, 0.655, "TVD / low-$k$", "white", BLUE, 0.14)

    # Basis glyphs.
    for k, x in enumerate([0.14, 0.235, 0.33, 0.425]):
        rounded(ax, x - 0.033, 0.37, 0.066, 0.13, "white", RED, 1.0, 0.010)
        if k % 2 == 0:
            wave(ax, x - 0.026, 0.395, 0.052, 0.085, "high", RED, 1.4)
        else:
            wave(ax, x - 0.026, 0.395, 0.052, 0.085, "sine_high", RED, 1.25)
    for k, x in enumerate([0.55, 0.645, 0.74, 0.835]):
        rounded(ax, x - 0.033, 0.37, 0.066, 0.13, "white", BLUE, 1.0, 0.010)
        wave(ax, x - 0.026, 0.395, 0.052, 0.085, "low" if k % 2 == 0 else "sine_low", BLUE, 1.4)

    # Edge-function nodes and cross-channel fusion.
    low_nodes = [(x, 0.59) for x in [0.55, 0.645, 0.74, 0.835]]
    high_nodes = [(x, 0.59) for x in [0.14, 0.235, 0.33, 0.425]]
    out_nodes = [(0.37, 0.82), (0.50, 0.84), (0.63, 0.82)]
    for p in high_nodes + low_nodes + out_nodes:
        ax.add_patch(Circle(p, 0.022, facecolor="white", edgecolor=INK, lw=1.2, zorder=6))
    for i, a in enumerate(high_nodes):
        for j, b in enumerate(out_nodes):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=0.75, alpha=0.72, zorder=2)
    for i, a in enumerate(low_nodes):
        for j, b in enumerate(out_nodes):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=0.75, alpha=0.72, zorder=2)
    for x in [0.14, 0.235, 0.33, 0.425, 0.55, 0.645, 0.74, 0.835]:
        ax.plot([x, x], [0.50, 0.568], color=RED if x < 0.48 else BLUE, lw=1.0, alpha=0.8)

    txt(ax, 0.275, 0.535, r"$\tanh[a(s-b)]$", 7.3, RED, "bold")
    txt(ax, 0.685, 0.535, r"$\sum_k c_k B_k(s)$", 7.3, BLUE, "bold")

    # Input routing: physics chooses the branch, coordinate reaches both.
    arr(ax, (0.12, 0.17), (0.20, 0.31), RED, 1.2, 10)
    arr(ax, (0.33, 0.17), (0.33, 0.31), RED, 1.2, 10)
    arr(ax, (0.12, 0.17), (0.61, 0.31), BLUE, 1.2, 10, rad=-0.15)
    arr(ax, (0.54, 0.17), (0.72, 0.31), BLUE, 1.2, 10)
    arr(ax, (0.54, 0.17), (0.42, 0.31), RED, 1.2, 10)

    # Output gate and candidate profile.
    arr(ax, (0.50, 0.86), (0.50, 0.94), MAGENTA, 1.7, 13)
    rounded(ax, 0.42, 0.925, 0.16, 0.055, MAGENTA_LIGHT, MAGENTA, 1.3, 0.018)
    txt(ax, 0.50, 0.952, r"$\beta_i=s_i\alpha_i$", 10.5, MAGENTA, "bold")
    arr(ax, (0.58, 0.952), (0.655, 0.952), MAGENTA, 1.4, 11)
    rounded(ax, 0.655, 0.925, 0.12, 0.055, "white", GREEN, 1.2, 0.018)
    txt(ax, 0.715, 0.952, r"$q^*_{i,q}$", 10, GREEN, "bold")
    txt(ax, 0.79, 0.952, r"$\sum_q\delta q=0$", 7.4, GREEN, "bold", ha="left")

    txt(ax, 0.022, 0.12, "inputs", 7, GREY, "bold", ha="left")
    txt(ax, 0.92, 0.12, "sensor + stencil", 7, GREY, "bold", ha="right")

    save(fig, "fig1_multiscale_feature_encoder")


def figure2_dkan_edge_functions():
    fig, ax = canvas((12.0, 5.15))

    txt(ax, 0.018, 0.955, "a", 11, INK, "bold", ha="left")
    txt(ax, 0.33, 0.925, "EDGE-FUNCTION GRAPH", 7.2, GREY, "bold")

    # Input function bank.
    for k, y in enumerate([0.73, 0.57, 0.41, 0.25]):
        rounded(ax, 0.055, y - 0.055, 0.095, 0.11, "white", GREY, 1.0, 0.010)
        wave(ax, 0.067, y - 0.037, 0.071, 0.078, ["spline", "tanh", "sine_high", "low"][k], [BLUE, RED, RED, BLUE][k], 1.35)
    ax.add_patch(Circle((0.025, 0.49), 0.023, facecolor="white", edgecolor=INK, lw=1.4))
    arr(ax, (0.048, 0.49), (0.055, 0.57), GREY, 1.1, 9)
    arr(ax, (0.048, 0.49), (0.055, 0.41), GREY, 1.1, 9)

    # DKAN network with colored edge functions.
    layers = [
        [(0.21, y) for y in [0.73, 0.57, 0.41, 0.25]],
        [(0.35, y) for y in [0.78, 0.64, 0.50, 0.36, 0.22]],
        [(0.50, y) for y in [0.71, 0.54, 0.37]],
        [(0.61, 0.54)],
    ]
    cols = [BLUE, RED, GOLD]
    for li in range(len(layers) - 1):
        for ia, a in enumerate(layers[li]):
            for ib, b in enumerate(layers[li + 1]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=cols[(ia + ib + li) % 3], lw=0.55, alpha=0.45, zorder=1)
    for li, layer in enumerate(layers):
        for p in layer:
            ax.add_patch(Circle(p, 0.018 if li < 3 else 0.027, facecolor="white" if li < 3 else MAGENTA_LIGHT, edgecolor=INK if li < 3 else MAGENTA, lw=1.2, zorder=6))
    txt(ax, 0.61, 0.54, r"$\alpha_i$", 9.3, MAGENTA, "bold")
    tag(ax, 0.34, 0.865, "trainable edge maps", "white", MAGENTA, 0.16)

    # Gate multiplication and basis selection.
    rounded(ax, 0.57, 0.79, 0.085, 0.075, MAGENTA_LIGHT, MAGENTA, 1.3, 0.016)
    txt(ax, 0.612, 0.827, r"$s_i$", 11, MAGENTA, "bold")
    arr(ax, (0.612, 0.79), (0.675, 0.60), MAGENTA, 1.4, 11)
    arr(ax, (0.638, 0.54), (0.675, 0.54), MAGENTA, 1.5, 12)
    ax.add_patch(Circle((0.70, 0.54), 0.026, facecolor="white", edgecolor=MAGENTA, lw=1.5, zorder=6))
    txt(ax, 0.70, 0.54, r"$\times$", 12, MAGENTA, "bold")
    txt(ax, 0.70, 0.48, r"$\beta_i$", 9.5, MAGENTA, "bold")
    arr(ax, (0.726, 0.54), (0.77, 0.54), MAGENTA, 1.6, 12)

    # Explicit MC/jump profiles.
    rounded(ax, 0.79, 0.62, 0.16, 0.22, GOLD_LIGHT, GOLD, 1.6, 0.025)
    wave(ax, 0.815, 0.675, 0.11, 0.13, "low", BLUE, 2.2)
    txt(ax, 0.87, 0.64, r"$q^{MC}$", 9, BLUE, "bold")
    rounded(ax, 0.79, 0.25, 0.16, 0.22, RED_LIGHT, RED, 1.6, 0.025)
    wave(ax, 0.815, 0.305, 0.11, 0.13, "high", RED, 2.2)
    txt(ax, 0.87, 0.27, r"$q^{jump}$", 9, RED, "bold")
    arr(ax, (0.77, 0.54), (0.79, 0.72), GOLD, 1.3, 10)
    arr(ax, (0.77, 0.54), (0.79, 0.36), RED, 1.3, 10)
    txt(ax, 0.775, 0.885, "b", 11, INK, "bold", ha="left")
    txt(ax, 0.87, 0.875, "BASIS PAIR", 7.2, GREY, "bold")

    # Edge-function decomposition callout, visually matching the supplied DKAN reference.
    rounded(ax, 0.19, 0.025, 0.52, 0.16, "#FFF9EA", GOLD, 1.2, 0.020)
    txt(ax, 0.175, 0.185, "c", 11, INK, "bold", ha="left", va="bottom")
    txt(ax, 0.45, 0.163, r"$\psi_e=\sum_k c_{ek}B_k+\omega_e\tanh[a_e(s-b_e)]$", 8.3, INK, "bold")
    wave(ax, 0.22, 0.045, 0.10, 0.075, "spline", BLUE, 1.8)
    txt(ax, 0.27, 0.034, "spline", 6.5, BLUE, "bold")
    txt(ax, 0.345, 0.085, "+", 14, INK, "bold")
    wave(ax, 0.38, 0.045, 0.10, 0.075, "tanh", RED, 1.8)
    txt(ax, 0.43, 0.034, "jump", 6.5, RED, "bold")
    txt(ax, 0.505, 0.085, "=", 14, INK, "bold")
    wave(ax, 0.54, 0.045, 0.13, 0.075, "blend", INK, 2.0)
    txt(ax, 0.605, 0.034, r"$\psi_e$", 6.8, INK, "bold")

    # The gate acts on candidate profiles, while the zero-mean correction preserves the parent mean.
    rounded(ax, 0.75, 0.025, 0.21, 0.16, GREEN_LIGHT, GREEN, 1.3, 0.020)
    wave(ax, 0.775, 0.075, 0.075, 0.075, "blend", MAGENTA, 1.9)
    txt(ax, 0.87, 0.135, r"$q^*=(1-\beta)q^{MC}+\beta q^{jump}$", 7.3, GREEN, "bold")
    txt(ax, 0.87, 0.072, r"$\sum_q(q^*_{i,q}-\bar q_i)=0$", 7.3, GREEN, "bold")
    ax.plot([0.81, 0.745, 0.745], [0.66, 0.52, 0.205], color=BLUE, lw=1.0, zorder=4)
    arr(ax, (0.745, 0.205), (0.785, 0.185), BLUE, 1.0, 9)
    arr(ax, (0.84, 0.25), (0.845, 0.185), RED, 1.0, 9, rad=-0.05)

    save(fig, "fig2_dkan_edge_function_subcell")


def _shield(ax, x, y, symbol, small):
    r = 0.045
    pts = np.array([[x-r*0.75,y+r*0.58],[x,y+r*0.88],[x+r*0.75,y+r*0.58],[x+r*0.63,y-r*0.22],[x,y-r*0.85],[x-r*0.63,y-r*0.22]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=GREEN_LIGHT, edgecolor=GREEN, lw=1.5, zorder=5))
    txt(ax, x, y, symbol, 7.2, GREEN, "bold")


def figure3_complete_model():
    fig, ax = canvas((12.2, 5.35))

    # Small offline strip: only the gate is trained.
    rounded(ax, 0.035, 0.79, 0.93, 0.16, VIOLET_LIGHT, VIOLET, 1.2, 0.025)
    rounded(ax, 0.07, 0.835, 0.13, 0.07, "white", VIOLET, 1.0, 0.014)
    txt(ax, 0.135, 0.87, "WENO", 9, VIOLET, "bold")
    arr(ax, (0.20, 0.87), (0.31, 0.87), VIOLET, 1.4, 11)
    wave(ax, 0.23, 0.835, 0.055, 0.07, "high", RED, 1.6)
    rounded(ax, 0.31, 0.835, 0.14, 0.07, "white", VIOLET, 1.0, 0.014)
    txt(ax, 0.38, 0.87, r"$\beta_i^{oracle}$", 9.5, VIOLET, "bold")
    arr(ax, (0.45, 0.87), (0.57, 0.87), VIOLET, 1.4, 11)
    rounded(ax, 0.57, 0.825, 0.15, 0.09, MAGENTA_LIGHT, MAGENTA, 1.3, 0.018)
    txt(ax, 0.645, 0.87, "DKAN", 10, MAGENTA, "bold")
    arr(ax, (0.72, 0.87), (0.86, 0.87), VIOLET, 1.4, 11, style="<|-|>")
    rounded(ax, 0.86, 0.835, 0.075, 0.07, "white", VIOLET, 1.0, 0.014)
    txt(ax, 0.898, 0.87, r"$\mathcal{L}_g$", 10, VIOLET, "bold")
    txt(ax, 0.05, 0.935, "OFFLINE", 8, VIOLET, "bold", ha="left")
    txt(ax, 0.022, 0.952, "a", 11, INK, "bold", ha="left")
    txt(ax, 0.775, 0.935, "oracle supervision", 7, VIOLET, "bold")

    # Online conservative backbone.
    txt(ax, 0.05, 0.725, "ONLINE", 8, BLUE, "bold", ha="left")
    txt(ax, 0.022, 0.735, "b", 11, INK, "bold", ha="left")
    stages = [
        (0.045, 0.48, 0.13, 0.16, BLUE_LIGHT, BLUE, "cells"),
        (0.205, 0.48, 0.13, 0.16, BLUE_LIGHT, BLUE, "HLLC"),
        (0.365, 0.48, 0.13, 0.16, BLUE_LIGHT, BLUE, r"$\bar{\mathbf{U}}^{n+1}$"),
        (0.525, 0.48, 0.13, 0.16, MAGENTA_LIGHT, MAGENTA, "DKAN"),
        (0.685, 0.48, 0.13, 0.16, GOLD_LIGHT, GOLD, r"$q^*$"),
    ]
    for x, y, w, h, fc, ec, lab in stages:
        rounded(ax, x, y, w, h, fc, ec, 1.5, 0.022)
        txt(ax, x+w/2, y+h/2, lab, 10, ec, "bold")
    for a, b, col in [(0.175,0.205,BLUE),(0.335,0.365,BLUE),(0.495,0.525,MAGENTA),(0.655,0.685,GOLD)]:
        arr(ax, (a,0.56),(b,0.56),col,1.7,13)

    lock_icon(ax, 0.226, 0.615, BLUE, 0.9)
    txt(ax, 0.270, 0.675, "fixed flux", 6.8, BLUE, "bold")
    lock_icon(ax, 0.385, 0.615, BLUE, 0.9)
    tag(ax, 0.59, 0.655, r"frozen $\theta_g$", "white", MAGENTA, 0.12)
    txt(ax, 0.67, 0.605, r"$\beta_i=s_i\alpha_i$", 6.8, MAGENTA, "bold")

    # Offline checkpoint is reused online; the FV time integrator remains untouched.
    arr(ax, (0.645, 0.825), (0.61, 0.65), VIOLET, 1.1, 10, rad=0.10, ls="--")

    # Cell and flux glyphs.
    for k, hh in enumerate([0.72, 0.76, 0.32, 0.27]):
        ax.add_patch(Rectangle((0.06+0.026*k,0.505),0.022,0.09*hh,facecolor=BLUE,edgecolor="none",alpha=0.72,zorder=5))
    arr(ax, (0.225,0.585),(0.315,0.585),BLUE,1.3,10)
    txt(ax, 0.27, 0.615, r"$\hat F_{i+1/2}$", 7.5, BLUE, "bold")
    txt(ax, 0.27, 0.465, "shared interface flux", 6.8, BLUE, "bold")
    # DKAN micro-network.
    for a in [(0.55,0.59),(0.55,0.53)]:
        for b in [(0.60,0.60),(0.60,0.56),(0.60,0.52)]:
            ax.plot([a[0],b[0]],[a[1],b[1]],color=MAGENTA,lw=0.7,alpha=0.6,zorder=5)
    for p in [(0.55,0.59),(0.55,0.53),(0.60,0.60),(0.60,0.56),(0.60,0.52)]:
        ax.add_patch(Circle(p,0.008,facecolor="white",edgecolor=MAGENTA,lw=0.8,zorder=6))
    wave(ax,0.705,0.51,0.09,0.10,"blend",MAGENTA,2.0)

    # Sensor and basis branches into DKAN/candidate.
    rounded(ax, 0.515, 0.30, 0.15, 0.09, "white", MAGENTA, 1.2, 0.016)
    txt(ax, 0.59, 0.345, r"$s_i=\max(|\Delta p|,[-\Delta u]_+,C_i)$", 7.8, MAGENTA, "bold")
    arr(ax, (0.59,0.39),(0.59,0.48),MAGENTA,1.4,11)
    rounded(ax, 0.685, 0.29, 0.13, 0.10, "white", GOLD, 1.2, 0.016)
    wave(ax,0.700,0.315,0.045,0.06,"low",BLUE,1.5)
    wave(ax,0.755,0.315,0.045,0.06,"high",RED,1.5)
    arr(ax,(0.75,0.39),(0.75,0.48),GOLD,1.4,11)

    # Hard trust filters and accepted subcells.
    arr(ax,(0.815,0.56),(0.845,0.56),GREEN,1.6,12)
    txt(ax, 0.835, 0.68, "c", 11, INK, "bold", ha="left")
    txt(ax, 0.905, 0.68, "TRUST STACK", 7.2, GREEN, "bold")
    _shield(ax,0.865,0.56,r"$[q_{min},q_{max}]$","stencil")
    arr(ax,(0.905,0.56),(0.925,0.56),GREEN,1.3,10)
    _shield(ax,0.945,0.56,r"$\rho,p>0$","positive")
    arr(ax,(0.945,0.51),(0.945,0.38),GREEN,1.5,12)
    _shield(ax,0.945,0.32,r"$TV\leq\tau$","trust")
    arr(ax,(0.905,0.32),(0.84,0.32),GREEN,1.5,12)
    for k,hh in enumerate([0.72,0.70,0.63,0.27,0.24]):
        ax.add_patch(Rectangle((0.835+0.026*k,0.18),0.022,0.09*hh,facecolor="#78B47F",edgecolor=GREEN,lw=0.7,zorder=5))
    arr(ax,(0.885,0.29),(0.885,0.255),GREEN,1.5,11)
    txt(ax, 0.855, 0.275, "accept", 6.8, GREEN, "bold", ha="right")
    txt(ax,0.91,0.115,r"$\frac{1}{Q}\sum_q\widetilde{\mathbf{U}}_{i,q}=\bar{\mathbf{U}}_i$",8.0,GREEN,"bold")

    # Any unsafe proposal falls back to the fixed monotone reconstruction.
    rounded(ax, 0.735, 0.075, 0.095, 0.095, LIGHT, GREY, 1.1, 0.016)
    wave(ax, 0.75, 0.095, 0.065, 0.055, "low", BLUE, 1.5)
    txt(ax, 0.783, 0.086, "MC", 6.8, GREY, "bold")
    arr(ax, (0.945, 0.275), (0.795, 0.17), RED, 1.1, 10, rad=0.20, ls="--")
    txt(ax, 0.805, 0.245, "reject", 6.8, RED, "bold")

    # Algebraic conservation strip.
    rounded(ax,0.045,0.09,0.66,0.10,"#EEF5FA",BLUE,1.1,0.018)
    txt(ax,0.375,0.14,r"$\bar{\mathbf{U}}_i^{n+1}=\bar{\mathbf{U}}_i^n-\frac{\Delta t}{\Delta x}(\hat{\mathbf{F}}_{i+1/2}-\hat{\mathbf{F}}_{i-1/2})$",9.5,BLUE,"bold")

    save(fig, "fig3_complete_cse_dkan_fv")


if __name__ == "__main__":
    figure1_multiscale_encoder()
    figure2_dkan_edge_functions()
    figure3_complete_model()
    print("wrote revised reference-style schematic set")
