"""Draw three concise vector schematics for the CSE-DKAN-FV manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path("paper_build/latex/figures")

INK = "#17212B"
MUTED = "#66717C"
FV = "#DCEAF7"
FV_EDGE = "#3D78A8"
DKAN = "#F1DDE5"
DKAN_EDGE = "#A95774"
BASE = "#F5E8CF"
BASE_EDGE = "#C18732"
SAFE = "#DDEEDC"
SAFE_EDGE = "#4C8B58"
SOFT = "#F4F6F8"
RED = "#C44E52"
BLUE = "#2E6FAD"
GOLD = "#D3942B"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "mathtext.fontset": "dejavusans",
    }
)


def canvas(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def box(ax, xy, w, h, fc, ec, radius=0.025, lw=1.5, z=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, a, b, color=INK, lw=1.6, scale=13, style="-|>", z=4, connection=None):
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        connectionstyle=connection,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=9, weight="normal", color=INK, ha="center", va="center", z=8):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)


def profile(ax, x0, y0, w, h, kind, color, lw=2.0, z=6):
    x = np.linspace(0, 1, 160)
    if kind == "mc":
        y = 0.77 - 0.50 / (1 + np.exp(-(x - 0.52) * 12))
    elif kind == "jump":
        y = np.where(x < 0.53, 0.76, 0.24)
    elif kind == "blend":
        y = 0.77 - 0.52 / (1 + np.exp(-(x - 0.53) * 42))
        y += 0.025 * np.sin(7 * np.pi * x) * np.exp(-((x - 0.53) / 0.20) ** 2)
    elif kind == "smooth":
        y = 0.50 + 0.22 * np.sin(2 * np.pi * x)
    else:
        raise ValueError(kind)
    ax.plot(x0 + w * x, y0 + h * y, color=color, lw=lw, solid_capstyle="round", zorder=z)
    ax.plot([x0, x0 + w], [y0 + 0.08 * h, y0 + 0.08 * h], color="#BCC4CC", lw=0.7, zorder=z - 1)


def shield(ax, center, r, fc, ec, symbol, subtitle=None, symbol_size=10):
    cx, cy = center
    pts = np.array(
        [
            [cx - 0.72 * r, cy + 0.62 * r],
            [cx, cy + 0.90 * r],
            [cx + 0.72 * r, cy + 0.62 * r],
            [cx + 0.62 * r, cy - 0.25 * r],
            [cx, cy - 0.88 * r],
            [cx - 0.62 * r, cy - 0.25 * r],
        ]
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=4))
    label(ax, cx, cy + 0.08 * r, symbol, size=symbol_size, weight="bold", color=ec)
    if subtitle:
        label(ax, cx, cy - 0.37 * r, subtitle, size=6.5, color=ec)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 360},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def figure_architecture():
    fig, ax = canvas((12.2, 5.7))

    # Layer 1: conservative parent-cell evolution.
    label(ax, 0.03, 0.93, "A", 11, "bold", FV_EDGE, ha="left")
    label(ax, 0.07, 0.93, "CONSERVATIVE PARENT CELLS", 10, "bold", INK, ha="left")
    label(ax, 0.96, 0.93, r"$\bar{\mathbf{U}}^{n}\;\longrightarrow\;\bar{\mathbf{U}}^{n+1}$", 11, "bold", FV_EDGE, ha="right")

    x0, y0, cw, ch = 0.075, 0.64, 0.13, 0.17
    heights = [0.72, 0.67, 0.76, 0.33, 0.28, 0.24]
    for j, hval in enumerate(heights):
        x = x0 + j * cw
        fc = "#C8DFF2" if j == 2 else FV
        ec = BLUE if j == 2 else FV_EDGE
        ax.add_patch(Rectangle((x, y0), cw, ch, facecolor=fc, edgecolor=ec, lw=1.2, zorder=2))
        ax.add_patch(Rectangle((x + 0.018, y0 + 0.02), cw - 0.036, ch * hval, facecolor="#76A7CC", edgecolor="none", alpha=0.55, zorder=3))
        label(ax, x + cw / 2, y0 - 0.032, rf"$\bar{{U}}_{{{j-2}}}$", 8, color=MUTED)
        if j > 0:
            xb = x
            arrow(ax, (xb - 0.032, y0 + ch + 0.055), (xb + 0.032, y0 + ch + 0.055), FV_EDGE, 1.35, 11)
            label(ax, xb, y0 + ch + 0.092, rf"$\hat F_{{{j}-1/2}}$", 7.5, color=FV_EDGE)
    box(ax, (0.82, 0.655), 0.13, 0.12, FV, FV_EDGE)
    label(ax, 0.885, 0.715, "HLLC", 11, "bold", FV_EDGE)
    arrow(ax, (0.79, 0.715), (0.82, 0.715), FV_EDGE, 1.7, 13)
    label(ax, 0.81, 0.80, "shared flux", 7.5, color=MUTED)
    arrow(ax, (0.40, 0.635), (0.40, 0.535), FV_EDGE, 1.8, 14)

    # Layer 2: local learned reconstruction.
    label(ax, 0.03, 0.49, "B", 11, "bold", DKAN_EDGE, ha="left")
    label(ax, 0.07, 0.49, "LOCAL SUBCELL RECONSTRUCTION", 10, "bold", INK, ha="left")

    # Local stencil glyph.
    box(ax, (0.055, 0.17), 0.16, 0.22, SOFT, "#9BA7B2")
    for k, hh in enumerate([0.42, 0.62, 0.82, 0.35, 0.30]):
        xx = 0.073 + 0.027 * k
        ax.add_patch(Rectangle((xx, 0.225), 0.018, 0.15 * hh, facecolor=FV_EDGE, edgecolor="none", alpha=0.72, zorder=5))
    arrow(ax, (0.082, 0.345), (0.188, 0.345), MUTED, 1.0, 9, style="<->")
    label(ax, 0.135, 0.195, r"$\bar{\mathbf{U}}_{i-2:i+2}$", 8.5, "bold")

    # Sensor + DKAN gate.
    box(ax, (0.255, 0.17), 0.17, 0.22, DKAN, DKAN_EDGE)
    for xi, yy in [(0.285, 0.31), (0.285, 0.25), (0.285, 0.20)]:
        ax.add_patch(Circle((xi, yy), 0.012, facecolor="white", edgecolor=DKAN_EDGE, lw=1.2, zorder=5))
    for xi, yy in [(0.345, 0.325), (0.345, 0.27), (0.345, 0.215), (0.345, 0.18)]:
        ax.add_patch(Circle((xi, yy), 0.011, facecolor="#F8EDF1", edgecolor=DKAN_EDGE, lw=1.0, zorder=5))
    for a in [(0.285, 0.31), (0.285, 0.25), (0.285, 0.20)]:
        for b in [(0.345, 0.325), (0.345, 0.27), (0.345, 0.215), (0.345, 0.18)]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=DKAN_EDGE, lw=0.55, alpha=0.55, zorder=3)
    arrow(ax, (0.36, 0.26), (0.399, 0.26), DKAN_EDGE, 1.2, 10)
    label(ax, 0.342, 0.145, r"$s_i\times\alpha_i$", 8.5, "bold", DKAN_EDGE)
    arrow(ax, (0.216, 0.28), (0.255, 0.28), INK, 1.5, 12)

    # Two basis profiles and blend.
    box(ax, (0.465, 0.17), 0.20, 0.22, BASE, BASE_EDGE)
    profile(ax, 0.48, 0.27, 0.072, 0.105, "mc", BLUE, 1.7)
    profile(ax, 0.568, 0.27, 0.072, 0.105, "jump", RED, 1.7)
    label(ax, 0.516, 0.215, "MC", 8, "bold", BLUE)
    label(ax, 0.604, 0.215, "JUMP", 8, "bold", RED)
    label(ax, 0.56, 0.195, r"$(1-\beta_i)\,q^{MC}+\beta_i\,q^{jump}$", 8.1, "bold", BASE_EDGE)
    arrow(ax, (0.425, 0.28), (0.465, 0.28), INK, 1.5, 12)
    label(ax, 0.445, 0.315, r"$\beta_i$", 8.8, "bold", DKAN_EDGE)

    # Trust filters.
    arrow(ax, (0.665, 0.28), (0.705, 0.28), INK, 1.5, 12)
    shield(ax, (0.735, 0.28), 0.055, SAFE, SAFE_EDGE, r"$[\min,\max]$", "stencil", 8.4)
    arrow(ax, (0.777, 0.28), (0.802, 0.28), SAFE_EDGE, 1.4, 11)
    shield(ax, (0.832, 0.28), 0.055, SAFE, SAFE_EDGE, r"$\rho,p>0$", "positive", 8.8)
    arrow(ax, (0.874, 0.28), (0.899, 0.28), SAFE_EDGE, 1.4, 11)
    shield(ax, (0.929, 0.28), 0.055, SAFE, SAFE_EDGE, r"$TV\leq\tau$", "trust", 8.6)

    # Accepted subcells and exact mean.
    arrow(ax, (0.929, 0.225), (0.929, 0.135), SAFE_EDGE, 1.7, 13)
    for k, hh in enumerate([0.74, 0.72, 0.67, 0.28, 0.25, 0.24]):
        xx = 0.855 + 0.025 * k
        ax.add_patch(Rectangle((xx, 0.075), 0.023, 0.062 * hh, facecolor="#72AD78", edgecolor=SAFE_EDGE, lw=0.7, zorder=4))
    label(ax, 0.918, 0.048, r"$\frac{1}{Q}\sum_q\widetilde{\mathbf{U}}_{i,q}=\bar{\mathbf{U}}_i$", 7.6, "bold", SAFE_EDGE)

    # The conceptual separation is the hero statement.
    box(ax, (0.055, 0.015), 0.68, 0.070, "#EEF4FA", FV_EDGE, radius=0.018, lw=1.0, z=1)
    label(ax, 0.395, 0.050, "FV  ->  cell means       |       DKAN  ->  subcells", 10.0, "bold", INK)

    save(fig, "new_fig3_cse_dkan_fv_principle")


def figure_dkan():
    fig, ax = canvas((12.0, 5.6))

    label(ax, 0.035, 0.93, "LOCAL FEATURES", 10, "bold", INK, ha="left")
    label(ax, 0.405, 0.93, "EDGE-FUNCTION DKAN", 10, "bold", INK, ha="center")
    label(ax, 0.845, 0.93, "SUBCELL BASIS", 10, "bold", INK, ha="center")

    # Input icon cards.
    cards = [(0.045, 0.69, FV, FV_EDGE), (0.045, 0.45, BASE, BASE_EDGE), (0.045, 0.21, DKAN, DKAN_EDGE)]
    for x, y, fc, ec in cards:
        box(ax, (x, y), 0.16, 0.15, fc, ec, radius=0.022)
    # Stencil slopes.
    xs = np.linspace(0.065, 0.18, 5)
    vals = [0.35, 0.52, 0.75, 0.30, 0.26]
    for xx, vv in zip(xs, vals):
        ax.add_patch(Rectangle((xx, 0.712), 0.014, 0.115 * vv, facecolor=FV_EDGE, edgecolor="none", alpha=0.85, zorder=5))
    ax.plot([0.062, 0.185], [0.710, 0.710], color=FV_EDGE, lw=0.9, alpha=0.6)
    label(ax, 0.125, 0.705, r"$\Delta\bar{\mathbf{U}}$", 8.5, "bold", FV_EDGE)
    # Pressure jump.
    ax.plot([0.065, 0.115, 0.115, 0.185], [0.50, 0.50, 0.57, 0.57], color=BASE_EDGE, lw=2.1)
    arrow(ax, (0.088, 0.535), (0.158, 0.535), BASE_EDGE, 1.2, 10, style="<->")
    label(ax, 0.125, 0.465, r"$|\Delta p|$", 8.5, "bold", BASE_EDGE)
    # Compression/contact.
    arrow(ax, (0.070, 0.285), (0.115, 0.285), DKAN_EDGE, 1.8, 13)
    arrow(ax, (0.182, 0.285), (0.137, 0.285), DKAN_EDGE, 1.8, 13)
    ax.plot([0.125, 0.125], [0.245, 0.325], color=DKAN_EDGE, lw=1.6, ls="--")
    label(ax, 0.125, 0.225, r"$-\Delta u,\ |\Delta\rho|$", 8.5, "bold", DKAN_EDGE)

    # Input-to-network arrows.
    for yy in [0.765, 0.525, 0.285]:
        arrow(ax, (0.205, yy), (0.265, 0.53 + 0.18 * (yy - 0.525)), MUTED, 1.2, 10)

    # DKAN network.
    layers = [
        [(0.285, y) for y in [0.68, 0.55, 0.42, 0.29]],
        [(0.405, y) for y in [0.73, 0.61, 0.49, 0.37, 0.25]],
        [(0.535, y) for y in [0.66, 0.51, 0.36]],
        [(0.635, 0.51)],
    ]
    edge_colors = [FV_EDGE, DKAN_EDGE, BASE_EDGE]
    for li in range(len(layers) - 1):
        for ia, a in enumerate(layers[li]):
            for ib, b in enumerate(layers[li + 1]):
                color = edge_colors[(ia + ib + li) % len(edge_colors)]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.55, alpha=0.38, zorder=1)
    for li, layer in enumerate(layers):
        for x, y in layer:
            fc = "white" if li < 3 else DKAN
            ec = MUTED if li < 3 else DKAN_EDGE
            ax.add_patch(Circle((x, y), 0.018 if li < 3 else 0.027, facecolor=fc, edgecolor=ec, lw=1.2, zorder=5))
    label(ax, 0.635, 0.51, r"$\alpha_i$", 9, "bold", DKAN_EDGE)

    # Edge-function glyphs: one smooth spline and one localized jump component.
    insets = [(0.305, 0.80, "spline", FV_EDGE), (0.455, 0.16, "tanh", DKAN_EDGE)]
    for x, y, kind, color in insets:
        box(ax, (x, y), 0.115, 0.085, "white", color, radius=0.012, lw=1.0, z=7)
        xx = np.linspace(0, 1, 80)
        if kind == "spline":
            yy = 0.47 + 0.23 * np.sin(2 * np.pi * xx) + 0.08 * np.sin(5 * np.pi * xx)
        else:
            yy = 0.50 + 0.28 * np.tanh(9 * (xx - 0.52))
        ax.plot(x + 0.012 + 0.09 * xx, y + 0.01 + 0.065 * yy, color=color, lw=1.5, zorder=8)
    label(ax, 0.435, 0.842, r"$+\;\omega_e\tanh[a_e(s-b_e)]$", 8.3, "bold", DKAN_EDGE, ha="left")
    label(ax, 0.305, 0.895, r"$\sum_k c_{ek}B_k(s)$", 8.2, "bold", FV_EDGE, ha="left")

    # Sensor multiplication.
    arrow(ax, (0.663, 0.51), (0.705, 0.51), DKAN_EDGE, 1.6, 12)
    ax.add_patch(Circle((0.73, 0.51), 0.026, facecolor="white", edgecolor=DKAN_EDGE, lw=1.5, zorder=5))
    label(ax, 0.73, 0.51, r"$\times$", 12, "bold", DKAN_EDGE)
    arrow(ax, (0.73, 0.64), (0.73, 0.54), DKAN_EDGE, 1.5, 12)
    box(ax, (0.683, 0.68), 0.094, 0.075, DKAN, DKAN_EDGE, radius=0.015)
    label(ax, 0.73, 0.718, r"$s_i$", 11, "bold", DKAN_EDGE)
    arrow(ax, (0.756, 0.51), (0.79, 0.51), DKAN_EDGE, 1.6, 12)
    label(ax, 0.785, 0.56, r"$\beta_i$", 9.5, "bold", DKAN_EDGE)

    # Basis blend.
    box(ax, (0.805, 0.59), 0.155, 0.20, BASE, BASE_EDGE)
    profile(ax, 0.823, 0.635, 0.12, 0.115, "mc", BLUE, 2.0)
    label(ax, 0.882, 0.61, r"$q^{MC}$", 8.5, "bold", BLUE)
    box(ax, (0.805, 0.28), 0.155, 0.20, "#F7E3E3", RED)
    profile(ax, 0.823, 0.325, 0.12, 0.115, "jump", RED, 2.0)
    label(ax, 0.882, 0.30, r"$q^{jump}$", 8.5, "bold", RED)
    arrow(ax, (0.79, 0.51), (0.805, 0.69), BASE_EDGE, 1.3, 10)
    arrow(ax, (0.79, 0.51), (0.805, 0.38), RED, 1.3, 10)

    # Candidate profile hero at bottom.
    box(ax, (0.285, 0.03), 0.675, 0.12, SOFT, "#9AA6B2", radius=0.018, lw=1.1)
    profile(ax, 0.705, 0.052, 0.150, 0.075, "blend", DKAN_EDGE, 2.4)
    label(ax, 0.315, 0.09, r"$q_{i,q}^{*}=(1-\beta_i)q_{i,q}^{MC}+\beta_i q_{i,q}^{jump}$", 11, "bold", INK, ha="left")
    label(ax, 0.945, 0.09, r"$\sum_q(q_{i,q}^{*}-\bar q_i)=0$", 8.0, "bold", SAFE_EDGE, ha="right")

    save(fig, "new_fig4_dkan_subcell_mechanism")


def figure_constraints():
    fig, ax = canvas((12.0, 5.4))

    label(ax, 0.03, 0.92, "DISCONTINUITY SENSOR", 10, "bold", INK, ha="left")
    label(ax, 0.40, 0.92, "LEARNED GATE", 10, "bold", INK, ha="center")
    label(ax, 0.73, 0.92, "TRUST FILTERS", 10, "bold", INK, ha="center")

    # Three sensor channels.
    sensor_y = [0.73, 0.52, 0.31]
    sensor_fc = [BASE, DKAN, FV]
    sensor_ec = [BASE_EDGE, DKAN_EDGE, FV_EDGE]
    sensor_sym = [r"$|\Delta p|$", r"$[-\Delta u]_+$", r"$|\Delta\rho|,\ |\Delta p|\approx0$"]
    for yy, fc, ec, sym in zip(sensor_y, sensor_fc, sensor_ec, sensor_sym):
        box(ax, (0.045, yy - 0.065), 0.18, 0.13, fc, ec, radius=0.022)
        label(ax, 0.135, yy, sym, 10, "bold", ec)
        arrow(ax, (0.225, yy), (0.275, 0.52), ec, 1.4, 11)
    ax.add_patch(Circle((0.305, 0.52), 0.043, facecolor="white", edgecolor=DKAN_EDGE, lw=1.6, zorder=5))
    label(ax, 0.305, 0.52, "max", 8.5, "bold", DKAN_EDGE)
    label(ax, 0.305, 0.45, r"$s_i\in[0,1]$", 9, "bold", DKAN_EDGE)

    # Learned gate with mini network.
    arrow(ax, (0.348, 0.52), (0.385, 0.52), DKAN_EDGE, 1.7, 12)
    box(ax, (0.385, 0.38), 0.16, 0.28, DKAN, DKAN_EDGE)
    nodes = [[(0.415, y) for y in [0.57, 0.48]], [(0.465, y) for y in [0.60, 0.52, 0.44]], [(0.515, 0.52)]]
    for a in nodes[0]:
        for b in nodes[1]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=DKAN_EDGE, lw=0.7, alpha=0.6)
    for a in nodes[1]:
        ax.plot([a[0], nodes[2][0][0]], [a[1], nodes[2][0][1]], color=DKAN_EDGE, lw=0.7, alpha=0.6)
    for layer in nodes:
        for p in layer:
            ax.add_patch(Circle(p, 0.010, facecolor="white", edgecolor=DKAN_EDGE, lw=1.0, zorder=5))
    label(ax, 0.465, 0.405, r"$\alpha_i^{DKAN}$", 9.5, "bold", DKAN_EDGE)
    arrow(ax, (0.545, 0.52), (0.585, 0.52), DKAN_EDGE, 1.7, 12)
    ax.add_patch(Circle((0.61, 0.52), 0.027, facecolor="white", edgecolor=DKAN_EDGE, lw=1.6, zorder=5))
    label(ax, 0.61, 0.52, r"$\times$", 12, "bold", DKAN_EDGE)
    arrow(ax, (0.61, 0.69), (0.61, 0.55), DKAN_EDGE, 1.4, 11)
    label(ax, 0.61, 0.72, r"$s_i$", 10, "bold", DKAN_EDGE)
    arrow(ax, (0.637, 0.52), (0.67, 0.52), DKAN_EDGE, 1.7, 12)
    label(ax, 0.655, 0.57, r"$\beta_i$", 10, "bold", DKAN_EDGE)

    # Candidate profile.
    box(ax, (0.655, 0.39), 0.11, 0.26, "#F8EFF2", DKAN_EDGE)
    profile(ax, 0.670, 0.455, 0.080, 0.13, "blend", DKAN_EDGE, 2.2)
    label(ax, 0.710, 0.415, r"$q^*$", 10, "bold", DKAN_EDGE)

    # Trust-filter portals.
    arrow(ax, (0.765, 0.52), (0.797, 0.52), INK, 1.5, 11)
    shield(ax, (0.82, 0.52), 0.058, SAFE, SAFE_EDGE, r"$[q_{min},q_{max}]$", "stencil", 7.5)
    arrow(ax, (0.865, 0.52), (0.885, 0.52), SAFE_EDGE, 1.3, 10)
    shield(ax, (0.91, 0.52), 0.058, SAFE, SAFE_EDGE, r"$\rho,p>0$", "positive", 8.5)
    arrow(ax, (0.91, 0.465), (0.91, 0.355), SAFE_EDGE, 1.5, 12)
    shield(ax, (0.91, 0.285), 0.064, SAFE, SAFE_EDGE, r"$TV\leq1.02\,TV_{MC}$", "trust", 7.8)

    # Accepted versus safe fallback.
    arrow(ax, (0.862, 0.285), (0.79, 0.285), SAFE_EDGE, 1.5, 12)
    box(ax, (0.655, 0.22), 0.135, 0.13, SAFE, SAFE_EDGE, radius=0.022)
    profile(ax, 0.673, 0.245, 0.098, 0.082, "blend", SAFE_EDGE, 2.2)
    label(ax, 0.722, 0.195, "ACCEPT", 8.2, "bold", SAFE_EDGE)

    arrow(ax, (0.71, 0.39), (0.60, 0.18), RED, 1.15, 10, connection="arc3,rad=0.10")
    ax.add_patch(Circle((0.655, 0.285), 0.017, facecolor="white", edgecolor=RED, lw=1.2, zorder=6))
    label(ax, 0.655, 0.285, r"$\times$", 8.5, "bold", RED)
    box(ax, (0.47, 0.08), 0.15, 0.12, SOFT, "#8A949E", radius=0.022)
    profile(ax, 0.487, 0.105, 0.115, 0.075, "mc", BLUE, 2.0)
    label(ax, 0.545, 0.055, "MC FALLBACK", 8.2, "bold", MUTED)

    # Compact invariant strip.
    box(ax, (0.045, 0.055), 0.34, 0.13, "#EEF4FA", FV_EDGE, radius=0.020, lw=1.1)
    label(ax, 0.095, 0.122, r"$\sum_i(\hat F_{i+1/2}-\hat F_{i-1/2})=0$", 9, "bold", FV_EDGE, ha="left")
    label(ax, 0.095, 0.079, r"$\frac{1}{Q}\sum_q\widetilde{\mathbf{U}}_{i,q}=\bar{\mathbf{U}}_i$", 9, "bold", SAFE_EDGE, ha="left")
    ax.add_patch(Circle((0.072, 0.122), 0.014, facecolor=FV_EDGE, edgecolor="none"))
    ax.add_patch(Circle((0.072, 0.079), 0.014, facecolor=SAFE_EDGE, edgecolor="none"))

    save(fig, "new_fig5_sensor_trust_logic")


if __name__ == "__main__":
    figure_architecture()
    figure_dkan()
    figure_constraints()
    print("wrote three CSE-DKAN-FV schematics")
