"""Create publication figures for the CSE-DKAN-FV manuscript using Python only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["xtick.labelsize"] = 6.5
plt.rcParams["ytick.labelsize"] = 6.5
plt.rcParams["legend.fontsize"] = 6.2
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


COLORS = {
    "MLP-PINN": "#B9BDC6",
    "GW-PINN": "#8FA4BF",
    "AV-PINN": "#D6A56A",
    "CI-PINN": "#9B86B8",
    "DKAN-PINN": "#4A8C95",
    "Fixed FV reconstruction": "#3E6D9C",
    "CSE-DKAN-FV": "#B64342",
    "Exact": "#161616",
    "Fine HLLC": "#294E75",
    "Piecewise constant": "#C7CBD1",
    "Fixed MC": "#8B8B8B",
    "Narrow DKAN-FV": "#C98786",
    "TV-trusted CSE-DKAN-FV": "#DD8A2E",
    "FNO": "#4A9D8F",
}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def box(ax, xy, width, height, text, face, edge="#3C3C3C", fontsize=6.5, lw=0.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#555555", lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.015, 0.59), 0.97, 0.37, facecolor="#F8FAFC", edgecolor="#AEB7C3", linewidth=0.9))
    ax.add_patch(Rectangle((0.015, 0.07), 0.97, 0.46, facecolor="#FFFDFC", edgecolor="#C6B9B5", linewidth=0.9))
    ax.text(0.03, 0.925, "OFFLINE: train the gate, not the time integrator", fontweight="bold", fontsize=8)
    ax.text(0.03, 0.495, "ONLINE: conservative evolution and constrained subcell inference", fontweight="bold", fontsize=8)

    train_nodes = [
        (0.04, 0.69, 0.15, "Exact/WENO\nlocal records", "#E7EEF6"),
        (0.25, 0.69, 0.17, "MC and jump\nbasis profiles", "#F3E8D5"),
        (0.48, 0.69, 0.16, "oracle blend\n" + r"$\alpha_i^\star$", "#E9E3F2"),
        (0.72, 0.69, 0.13, "DKAN gate\n" + r"$g_\theta(z_i)$", "#F1DCDD"),
    ]
    for idx, (x, y, w, label, face) in enumerate(train_nodes):
        box(ax, (x, y), w, 0.13, label, face, edge="#59616A")
        if idx < len(train_nodes) - 1:
            arrow(ax, (x + w, y + 0.065), (train_nodes[idx + 1][0], y + 0.065))
    box(ax, (0.70, 0.865), 0.27, 0.06, r"weighted oracle MSE $\mathcal{L}_{\rm gate}$", "#E8E1F0", edge="#755A91", fontsize=6.2)
    ax.add_patch(FancyArrowPatch((0.835, 0.865), (0.79, 0.82), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, linestyle="--", color="#755A91"))
    ax.text(0.86, 0.77, "backpropagation", fontsize=5.7, color="#755A91", rotation=90, va="center")
    ax.text(0.50, 0.625, "Checkpoint selection uses held-out subcell-profile MSE.", ha="center", fontsize=6.1, color="#5E4A70")

    online_nodes = [
        (0.035, 0.24, 0.13, "Riemann data\n$U(x,0)$", "#EDF1F6"),
        (0.205, 0.24, 0.15, "HLLC-MUSCL\nFV update", "#DCE8F4"),
        (0.395, 0.24, 0.14, "parent means\n" + r"$\bar U_i^{n+1}$", "#DCE8F4"),
        (0.575, 0.24, 0.12, "15 features\n$z_i$", "#E8EEF5"),
        (0.735, 0.24, 0.11, "DKAN\n" + r"$\alpha_i$", "#F1DCDD"),
        (0.885, 0.24, 0.09, "accepted\nsubcells", "#E3F0E4"),
    ]
    for idx, (x, y, w, label, face) in enumerate(online_nodes):
        box(ax, (x, y), w, 0.14, label, face, edge="#4B5965", fontsize=6.1)
        if idx < len(online_nodes) - 1:
            arrow(ax, (x + w, y + 0.07), (online_nodes[idx + 1][0], y + 0.07))
    ax.text(0.28, 0.13, r"$\bar U_i^{n+1}=\bar U_i^n-\frac{\Delta t}{\Delta x}(\hat F_{i+1/2}-\hat F_{i-1/2})$", ha="center", fontsize=6.4, color="#294E75")
    box(ax, (0.555, 0.095), 0.25, 0.075, r"$U_{i,q}^{\rm MC}\;\leftrightarrow\;U_{i,q}^{\rm jump}$", "#F7ECDD", edge="#B8813C", fontsize=6.2)
    arrow(ax, (0.68, 0.17), (0.79, 0.24), color="#B8813C")
    box(ax, (0.84, 0.095), 0.135, 0.075, "mean + stencil +\npositivity + TV", "#E4F0E4", edge="#4E8651", fontsize=5.7)
    arrow(ax, (0.907, 0.17), (0.93, 0.24), color="#4E8651")
    ax.text(0.50, 0.035, "Shared interface fluxes guarantee conservation; DKAN only refines the within-cell profile.", ha="center", fontsize=6.5, color="#244F75")
    save_figure(fig, output, "fig1_architecture")


def figure_dkan_mechanism(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    ax = axes[0]
    z = np.linspace(-1.2, 1.2, 300)
    dyt = 0.72 * np.tanh(3.1 * (z - 0.06))
    spline = 0.24 * np.exp(-((z + 0.58) / 0.22) ** 2) - 0.20 * np.exp(-((z - 0.45) / 0.18) ** 2)
    ax.plot(z, dyt, color="#B64342", lw=1.2, label="dynamic tanh")
    ax.plot(z, spline, color="#3E6D9C", lw=1.0, ls="--", label="B-spline sum")
    ax.plot(z, dyt + spline, color="#171717", lw=1.5, label=r"edge function $\phi(z)$")
    ax.axhline(0, color="#AAAAAA", lw=0.5)
    ax.set_xlabel("normalized feature z")
    ax.set_ylabel("edge response")
    ax.set_title("DKAN edge function")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for i, h in enumerate([0.24, 0.46, 0.72, 0.40, 0.18]):
        ax.add_patch(Rectangle((0.03 + 0.09 * i, 0.14), 0.065, h, facecolor="#CCDCEB", edgecolor="#557A9F", linewidth=0.7))
    ax.text(0.24, 0.05, "five-cell conservative context", ha="center", fontsize=6.2)
    for y in (0.25, 0.48, 0.71):
        ax.add_patch(plt.Circle((0.62, y), 0.045, facecolor="#F1DCDD", edgecolor="#A85250", linewidth=0.8))
        for x0 in (0.095, 0.185, 0.275, 0.365, 0.455):
            ax.plot([x0, 0.575], [0.3, y], color="#A8A8A8", lw=0.35)
    ax.add_patch(plt.Circle((0.88, 0.48), 0.055, facecolor="#E5F0E5", edgecolor="#4E8651", linewidth=0.8))
    for y in (0.25, 0.48, 0.71): ax.plot([0.665, 0.825], [y, 0.48], color="#8A6B6A", lw=0.55)
    ax.text(0.62, 0.88, "edge-wise spline + DyT", ha="center", fontsize=6.3)
    ax.text(0.88, 0.48, r"$\alpha_i$", ha="center", va="center", fontsize=8)
    ax.set_title("15 features to one bounded gate")
    panel_label(ax, "b")

    ax = axes[2]
    xi = np.linspace(-0.5, 0.5, 4)
    mc = 0.62 + 0.42 * xi
    jump = np.array([0.91, 0.83, 0.38, 0.36])
    jump += mc.mean() - jump.mean()
    blend = 0.30 * mc + 0.70 * jump
    ax.plot(xi, mc, marker="o", color=COLORS["Fixed MC"], lw=1.0, label="fixed MC")
    ax.plot(xi, jump, marker="s", color="#B8813C", lw=1.0, label="jump basis")
    ax.plot(xi, blend, marker="D", color=COLORS["CSE-DKAN-FV"], lw=1.3, label="accepted blend")
    ax.axhline(mc.mean(), color="#294E75", lw=0.8, ls=":", label="parent mean")
    ax.set_xlabel(r"subcell coordinate $\xi_q$")
    ax.set_ylabel("illustrative conservative state")
    ax.set_title("Zero-mean subcell blend")
    ax.legend(loc="best")
    panel_label(ax, "c")
    save_figure(fig, output, "fig2_dkan_mechanism")


def figure_constraint_logic(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        (0.05, 0.78, "DKAN proposal", "#F1DCDD", "resolution"),
        (0.28, 0.59, "stencil scaling", "#F4EBD8", "local bounds"),
        (0.51, 0.40, "positivity bisection", "#E6EFE0", r"$\rho,p>0$"),
        (0.74, 0.21, "density-TV trust", "#DDEBE4", "oscillation budget"),
    ]
    for i, (x, y, label, face, note) in enumerate(steps):
        box(ax, (x, y), 0.20, 0.12, label, face, fontsize=6.2)
        ax.text(x + 0.10, y - 0.045, note, ha="center", fontsize=5.6, color="#555555")
        if i < len(steps) - 1:
            arrow(ax, (x + 0.18, y), (steps[i + 1][0] + 0.02, steps[i + 1][1] + 0.12))
    ax.text(0.50, 0.035, "Each projection is a convex scaling toward fixed MC, so the parent mean is unchanged.", ha="center", fontsize=6.1)
    ax.set_title("Trust stack as a sequence of admissibility filters")
    panel_label(ax, "a")

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    items = [
        (0.06, 0.69, 0.88, 0.20, "Algebraic invariants", "shared flux conservation; exact subcell mean", "#DCE8F4", "#557A9F"),
        (0.12, 0.43, 0.76, 0.19, "Hard admissibility", "stencil bounds; positive density and pressure", "#E4F0E4", "#4E8651"),
        (0.20, 0.18, 0.60, 0.18, "Audits, not guarantees", "entropy; Rankine-Hugoniot; shock width; runtime", "#F7ECDD", "#B8813C"),
    ]
    for x, y, w, h, title, body, face, edge in items:
        box(ax, (x, y), w, h, "", face, edge=edge)
        ax.text(x + 0.02, y + h * 0.66, title, fontsize=6.6, fontweight="bold", color=edge)
        ax.text(x + 0.02, y + h * 0.30, body, fontsize=5.8)
    ax.set_title("What the architecture guarantees and what it only checks")
    panel_label(ax, "b")
    save_figure(fig, output, "fig3_constraint_logic")


def figure_canonical(source: Path, output: Path) -> None:
    data = pd.read_csv(source / "canonical_method_comparison.csv")
    methods = list(data[data.equation == "burgers"].method)
    colors = [COLORS[m] for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for row, equation in enumerate(("burgers", "euler")):
        subset = data[data.equation == equation].set_index("method").loc[methods]
        ax = axes[row, 0]
        values = subset.state_error.to_numpy(float)
        bars = ax.bar(np.arange(len(methods)), values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylabel("Normalized state error")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=32, ha="right")
        ax.set_title("Inviscid Burgers" if equation == "burgers" else "Sod Euler")
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper * 1.35)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.3f}", ha="center", va="bottom", fontsize=5.4, rotation=90)
        panel_label(ax, "a" if row == 0 else "c")

        ax = axes[row, 1]
        widths = subset.shock_width.to_numpy(float)
        valid = np.isfinite(widths)
        x = np.arange(len(methods))[valid]
        ax.bar(x, widths[valid], color=np.asarray(colors)[valid], edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylabel("10-90% shock width")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=32, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
        title = "Shock resolution (median of five seeds)"
        if equation == "euler":
            title += "\n(1 denotes failed width extraction)"
        ax.set_title(title)
        missing = np.flatnonzero(~valid)
        lower, _ = ax.get_ylim()
        for index in missing:
            ax.text(index, lower * 1.08, "n/a", ha="center", va="bottom", fontsize=5.5, color="#666666")
        panel_label(ax, "b" if row == 0 else "d")
    save_figure(fig, output, "fig9_canonical_baselines")


PROFILE_STYLES = {
    "fno_operator": ("fno", "FNO", COLORS["FNO"], "-", 0.85),
    "hllc_piecewise_constant": ("piecewise", "128-cell HLLC, PC", COLORS["Piecewise constant"], ":", 0.9),
    "hllc_fixed_mc_subcell": ("fixed_mc", "128-cell HLLC, MC", COLORS["Fixed MC"], "--", 0.95),
    "physics_gated_dkan_fv_sod_neighbourhood": ("legacy_dkan_fv", "DKAN-FV (narrow)", COLORS["Narrow DKAN-FV"], "-.", 0.9),
    "physics_gated_dkan_fv_engineering": ("dkan_fv", "CSE-DKAN-FV", COLORS["CSE-DKAN-FV"], "-", 1.25),
    "hllc_muscl_fine": ("fine_hllc", "512-cell HLLC", COLORS["Fine HLLC"], "--", 1.0),
}


def figure_casebook(source: Path, output: Path, cases: list[str], name: str, title: str) -> None:
    catalog = pd.read_csv(source / "casebook_catalog.csv").set_index("case_id")
    metrics = pd.read_csv(source / "casebook_metrics.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.85), constrained_layout=True)
    variables = (("density", r"Density $\rho$"), ("velocity", r"Velocity $u$"), ("pressure", r"Pressure $p$"))
    for row, case in enumerate(cases):
        data = pd.read_csv(source / f"profile_{case}.csv")
        x = data.x.to_numpy(float)
        for column, (variable, ylabel) in enumerate(variables):
            ax = axes[row, column]
            ax.plot(x, data[f"exact_{variable}"], color=COLORS["Exact"], lw=2.6, alpha=0.42, label="Exact", zorder=2)
            for method, (prefix, label, color, linestyle, linewidth) in PROFILE_STYLES.items():
                ax.plot(x, data[f"{prefix}_{variable}"], color=color, ls=linestyle, lw=linewidth, label=label, zorder=8 if method == "physics_gated_dkan_fv_engineering" else 4)
            ax.set_xlim(0, 1)
            ax.set_ylabel(ylabel)
            if row == 1: ax.set_xlabel("x")
            ax.grid(color="#E5E5E5", linewidth=0.4)
            panel_label(ax, chr(ord("a") + row * 4 + column))
        ax = axes[row, 3]
        block = metrics[metrics.case_id == case].set_index("method")
        order = list(PROFILE_STYLES)
        values = block.loc[order, "primitive_geometric_mean_l2"].to_numpy(float)
        y = np.arange(len(order))
        ax.barh(y, values, color=[PROFILE_STYLES[m][2] for m in order], height=0.68)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(max(values.min() * 0.72, 1.0e-4), values.max() * 1.65)
        for position, value in zip(y, values):
            ax.text(value * 1.06, position, f"{value:.3g}", va="center", ha="left", fontsize=5.0)
        ax.set_yticks(y)
        ax.set_yticklabels([PROFILE_STYLES[m][1] for m in order], fontsize=5.3)
        ax.invert_yaxis()
        ax.set_xlabel("Primary state error")
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.4, which="both")
        panel_label(ax, chr(ord("a") + row * 4 + 3))
        info = catalog.loc[case]
        axes[row, 0].text(0.00, 1.17, f"{case}: fixed/CSE gain {info.fixed_over_engineering_gain:.3f}; FNO/CSE gain {info.fno_over_engineering_gain:.2f}", transform=axes[row, 0].transAxes, fontsize=6.7, fontweight="bold", ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.055), fontsize=5.7)
    fig.suptitle(title, y=1.02, fontsize=8.2, fontweight="bold")
    save_figure(fig, output, name)


def figure_case_diagnostic(
    source: Path, output: Path, case_id: str, name: str
) -> None:
    """Draw one non-redundant, review-oriented plate for one Riemann problem.

    The hero atlas assigns every compared model a co-registered density result
    and absolute-error field.  The supporting quantitative panels replace the
    earlier compact casebook figure, so profiles are not repeated elsewhere.
    """
    catalog = pd.read_csv(source / "casebook_catalog.csv").set_index("case_id")
    metrics = pd.read_csv(source / "casebook_metrics.csv")
    profile = pd.read_csv(source / f"profile_{case_id}.csv")
    trajectory = pd.read_csv(source / f"trajectory_{case_id}.csv")
    losses = pd.read_csv(source / f"case_loss_{case_id}.csv")
    info = catalog.loc[case_id]

    times = np.sort(trajectory.time.unique())
    x = np.sort(trajectory.x.unique())
    shape = (len(times), len(x))
    exact_density = trajectory.exact_density.to_numpy(float).reshape(shape)
    blend = trajectory.cse_blend.to_numpy(float).reshape(shape)

    method_order = list(PROFILE_STYLES)
    density_fields = {
        method: trajectory[f"{method}_density"].to_numpy(float).reshape(shape)
        for method in method_order
    }
    error_fields = {
        method: trajectory[
            f"{method}_absolute_density_error"
        ].to_numpy(float).reshape(shape)
        for method in method_order
    }
    all_density = np.concatenate(
        [exact_density.ravel(), *[density_fields[m].ravel() for m in method_order]]
    )
    all_error = np.concatenate([error_fields[m].ravel() for m in method_order])
    density_min = float(np.min(all_density))
    density_max = float(np.max(all_density))
    error_max = max(float(np.quantile(all_error, 0.995)), 1.0e-10)
    error_min = max(error_max * 1.0e-4, 1.0e-8)
    error_norm = LogNorm(vmin=error_min, vmax=error_max, clip=True)
    contour_levels = np.linspace(
        float(exact_density.min()), float(exact_density.max()), 7
    )[1:-1]

    fig = plt.figure(figsize=(7.2, 8.55))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[3.55, 2.20], hspace=0.20,
        left=0.105, right=0.945, bottom=0.075, top=0.885,
    )
    atlas = outer[0].subgridspec(
        6, 4, width_ratios=[1.0, 0.028, 1.0, 0.028],
        hspace=0.055, wspace=0.12,
    )
    density_axes, error_axes = [], []
    density_image = error_image = None
    for row, method in enumerate(method_order):
        _, label, color, _, _ = PROFILE_STYLES[method]
        ax_result = fig.add_subplot(atlas[row, 0])
        ax_error = fig.add_subplot(atlas[row, 2])
        density_axes.append(ax_result)
        error_axes.append(ax_error)
        density_image = ax_result.pcolormesh(
            x, times, density_fields[method], shading="auto", cmap="cividis",
            vmin=density_min, vmax=density_max, rasterized=True,
        )
        # Exact contours provide the same spatial reference in every model row.
        if density_max > density_min:
            ax_result.contour(
                x, times, exact_density, levels=contour_levels,
                colors="#111111", linewidths=0.30, alpha=0.72,
            )
        error_image = ax_error.pcolormesh(
            x, times, np.maximum(error_fields[method], error_min),
            shading="auto", cmap="magma", norm=error_norm, rasterized=True,
        )
        ax_result.set_ylabel(label, color=color, fontsize=5.4, labelpad=4.0)
        ax_result.tick_params(axis="both", labelsize=5.1, length=2)
        ax_error.tick_params(axis="both", labelsize=5.1, length=2)
        ax_error.set_yticklabels([])
        if row < len(method_order) - 1:
            ax_result.set_xticklabels([])
            ax_error.set_xticklabels([])
        else:
            ax_result.set_xlabel("x")
            ax_error.set_xlabel("x")
        if row == 0:
            ax_result.set_title(
                r"Predicted density $\widehat{\rho}(x,t)$ with exact contours"
            )
            ax_error.set_title(
                r"Absolute density error $|\widehat{\rho}-\rho|$ (log colour)"
            )
            panel_label(ax_result, "a")
    density_cax = fig.add_subplot(atlas[:, 1])
    error_cax = fig.add_subplot(atlas[:, 3])
    density_bar = fig.colorbar(density_image, cax=density_cax)
    density_bar.set_label(r"density $\rho$")
    density_bar.ax.tick_params(labelsize=5.2)
    error_bar = fig.colorbar(error_image, cax=error_cax)
    error_bar.set_label(r"$|\widehat{\rho}-\rho|$")
    error_bar.ax.tick_params(labelsize=5.2)

    support = outer[1].subgridspec(2, 3, hspace=0.62, wspace=0.55)
    support_axes = np.asarray(
        [[fig.add_subplot(support[r, c]) for c in range(3)] for r in range(2)]
    )

    variables = (
        ("density", r"Density $\rho$"),
        ("velocity", r"Velocity $u$"),
        ("pressure", r"Pressure $p$"),
    )
    final_x = profile.x.to_numpy(float)
    for column, (variable, ylabel) in enumerate(variables):
        ax = support_axes[0, column]
        ax.plot(
            final_x, profile[f"exact_{variable}"], color=COLORS["Exact"],
            lw=2.8, alpha=0.38, label="Exact", zorder=2,
        )
        for method, (prefix, label, color, linestyle, linewidth) in PROFILE_STYLES.items():
            ax.plot(
                final_x, profile[f"{prefix}_{variable}"], color=color,
                ls=linestyle, lw=linewidth, label=label,
                zorder=8 if method == "physics_gated_dkan_fv_engineering" else 4,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Final-time {variable} profile")
        ax.grid(color="#E4E4E4", linewidth=0.4)
        panel_label(ax, chr(ord("b") + column))

    ax = support_axes[1, 0]
    for method, (_, label, color, linestyle, linewidth) in PROFILE_STYLES.items():
        block = losses[losses.method == method].sort_values("time_fraction")
        ax.plot(
            block.time_fraction, block.case_evaluation_loss,
            color=color, ls=linestyle, lw=max(linewidth, 0.9), label=label,
        )
    ax.set_yscale("log")
    ax.set_xlim(float(losses.time_fraction.min()), 1.0)
    ax.set_xlabel(r"normalized rollout time $t/t_f$")
    ax.set_ylabel(r"case evaluation loss $\mathcal{L}_{\mathrm{case}}(t)$")
    ax.set_title("Time-resolved six-model loss")
    ax.grid(color="#E1E1E1", linewidth=0.4, which="both")
    panel_label(ax, "e")

    ax = support_axes[1, 1]
    blend_image = ax.pcolormesh(
        x, times, blend, shading="auto", cmap="Blues", vmin=0.0,
        vmax=max(0.75, float(np.max(blend))), rasterized=True,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("physical time")
    ax.set_title("Median learned jump-blend gate")
    blend_bar = fig.colorbar(blend_image, ax=ax, fraction=0.045, pad=0.02)
    blend_bar.set_label(r"$\alpha$")
    blend_bar.ax.tick_params(labelsize=5.2)
    panel_label(ax, "f")

    ax = support_axes[1, 2]
    block = metrics[metrics.case_id == case_id].set_index("method").loc[method_order]
    component_matrix = block[
        ["density_normalized_l2", "velocity_normalized_l2", "pressure_normalized_l2"]
    ].to_numpy(float)
    heatmap = ax.imshow(
        np.log10(np.maximum(component_matrix, 1.0e-12)), aspect="auto",
        cmap="magma_r",
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r"$E_\rho$", r"$E_u$", r"$E_p$"])
    ax.set_yticks(np.arange(len(method_order)))
    short_labels = ["FNO", "HLLC-PC", "HLLC-MC", "DKAN-N", "CSE", "HLLC-512"]
    ax.set_yticklabels(short_labels, fontsize=4.7)
    ax.set_title(r"Final component errors, $\log_{10}$")
    fig.colorbar(heatmap, ax=ax, fraction=0.045, pad=0.02, label=r"$\log_{10} E$")
    panel_label(ax, "g")

    handles, labels = support_axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=7,
        bbox_to_anchor=(0.5, 0.938), fontsize=5.35,
    )
    fig.suptitle(
        f"{case_id} ({info.case_group.replace('_', ' ')}), "
        f"$t_f={info.final_time:.3f}$; fixed/CSE gain={info.fixed_over_engineering_gain:.3f}",
        y=0.982, fontsize=8.2, fontweight="bold",
    )
    save_figure(fig, output, name)


def figure_model_isolated_profiles(
    source: Path, output: Path, case_id: str, name: str
) -> None:
    """Give every comparator its own primitive-state result and error axes.

    The main case plate emphasizes space--time density localization.  This
    complementary appendix plate separates the six models so that a reviewer
    can inspect amplitude, phase, and oscillation errors in rho, u, and p
    without line occlusion.
    """
    profile = pd.read_csv(source / f"profile_{case_id}.csv")
    catalog = pd.read_csv(source / "casebook_catalog.csv").set_index("case_id")
    info = catalog.loc[case_id]
    x = profile.x.to_numpy(float)
    variables = (
        ("density", r"Density $\rho$"),
        ("velocity", r"Velocity $u$"),
        ("pressure", r"Pressure $p$"),
    )
    method_order = list(PROFILE_STYLES)

    error_limits: dict[str, tuple[float, float]] = {}
    for variable, _ in variables:
        exact = profile[f"exact_{variable}"].to_numpy(float)
        errors = np.concatenate(
            [
                np.abs(profile[f"{PROFILE_STYLES[m][0]}_{variable}"].to_numpy(float) - exact)
                for m in method_order
            ]
        )
        high = max(float(np.quantile(errors, 0.997)), float(np.max(errors)) * 0.25, 1.0e-8)
        low = max(high * 1.0e-5, 1.0e-10)
        error_limits[variable] = (low, high * 1.15)

    fig = plt.figure(figsize=(7.2, 8.65))
    outer = fig.add_gridspec(
        6, 3, left=0.10, right=0.975, bottom=0.055, top=0.925,
        hspace=0.32, wspace=0.24,
    )
    legend_handles = None
    for row, method in enumerate(method_order):
        prefix, label, color, linestyle, linewidth = PROFILE_STYLES[method]
        for column, (variable, title) in enumerate(variables):
            inner = outer[row, column].subgridspec(
                2, 1, height_ratios=[1.75, 0.85], hspace=0.04
            )
            ax_result = fig.add_subplot(inner[0])
            ax_error = fig.add_subplot(inner[1], sharex=ax_result)
            exact = profile[f"exact_{variable}"].to_numpy(float)
            predicted = profile[f"{prefix}_{variable}"].to_numpy(float)
            absolute = np.abs(predicted - exact)

            exact_line, = ax_result.plot(
                x, exact, color=COLORS["Exact"], lw=1.45, alpha=0.82,
                label="Exact",
            )
            prediction_line, = ax_result.plot(
                x, predicted, color=color, ls=linestyle,
                lw=max(linewidth, 1.0), label="Model result",
            )
            ax_result.fill_between(
                x, exact, predicted, color=color, alpha=0.12, linewidth=0,
            )
            error_line, = ax_error.plot(
                x, np.maximum(absolute, error_limits[variable][0]),
                color=color, lw=0.85, label="Absolute error",
            )
            ax_error.fill_between(
                x, error_limits[variable][0],
                np.maximum(absolute, error_limits[variable][0]),
                color=color, alpha=0.18, linewidth=0,
            )
            ax_error.set_yscale("log")
            ax_error.set_ylim(*error_limits[variable])
            ax_result.set_xlim(0.0, 1.0)
            ax_result.grid(color="#E6E6E6", linewidth=0.32)
            ax_error.grid(color="#E8E8E8", linewidth=0.28, which="both")
            ax_result.tick_params(axis="x", labelbottom=False, length=1.8)
            ax_result.tick_params(axis="y", labelsize=4.9, length=1.8)
            ax_error.tick_params(axis="both", labelsize=4.7, length=1.7)
            if row < len(method_order) - 1:
                ax_error.tick_params(axis="x", labelbottom=False)
            else:
                ax_error.set_xlabel("x", labelpad=1.0)
            if column == 0:
                ax_result.set_ylabel(label, color=color, fontsize=5.4, labelpad=3.5)
                ax_error.set_ylabel("abs. error", fontsize=4.8, labelpad=2.0)
            if row == 0:
                ax_result.set_title(title, fontsize=7.2, pad=3.0)
            if legend_handles is None:
                legend_handles = [exact_line, prediction_line, error_line]

    fig.legend(
        legend_handles, ["Exact", "model result", "pointwise absolute error"],
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.963), fontsize=5.9,
    )
    fig.suptitle(
        f"{case_id}: model-isolated primitive results and errors; "
        f"fixed/CSE gain={info.fixed_over_engineering_gain:.3f}",
        y=0.994, fontsize=8.1, fontweight="bold",
    )
    save_figure(fig, output, name)


def _history_envelope(data: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = data.pivot(index="iteration", columns="seed", values=value).sort_index()
    return pivot.index.to_numpy(float), pivot.median(axis=1).to_numpy(float), pivot.min(axis=1).to_numpy(float), pivot.max(axis=1).to_numpy(float)


def figure_training_and_case_error(source: Path, output: Path) -> None:
    history = pd.read_csv(source / "training_histories.csv")
    metrics = pd.read_csv(source / "casebook_metrics.csv")
    catalog = pd.read_csv(source / "casebook_catalog.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), constrained_layout=True)

    ax = axes[0, 0]
    fno = history[history.method == "FNO"]
    for value, label, color, linestyle in (("training_loss", "training weighted MSE", "#5D9C91", "-"), ("validation_loss", "validation MSE", "#245D55", "--")):
        x, median, low, high = _history_envelope(fno, value)
        ax.fill_between(x, low, high, color=color, alpha=0.12, lw=0)
        ax.plot(x, median, color=color, ls=linestyle, lw=1.15, label=label)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("recorded loss")
    ax.set_title("FNO family training (three seeds)"); ax.legend(); ax.grid(color="#E2E2E2", linewidth=0.45, which="both")
    panel_label(ax, "a")

    ax = axes[0, 1]
    for method, color, linestyle in (("CSE-DKAN-FV", COLORS["CSE-DKAN-FV"], "-"), ("CSE-DKAN-FV (narrow)", COLORS["Narrow DKAN-FV"], "--")):
        subset = history[history.method == method]
        xt, train_median, train_low, train_high = _history_envelope(subset, "training_loss")
        ax.fill_between(xt, train_low, train_high, color=color, alpha=0.07, lw=0)
        ax.plot(xt, train_median, color=color, ls=":", lw=0.9, alpha=0.78, label=method + " gate loss")
        x, median, low, high = _history_envelope(subset, "profile_validation_loss")
        ax.fill_between(x, low, high, color=color, alpha=0.12, lw=0)
        ax.plot(x, median, color=color, ls=linestyle, lw=1.2, label=method + " profile MSE")
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("recorded MSE")
    ax.set_title("DKAN training and checkpoint metrics"); ax.legend(fontsize=5.4); ax.grid(color="#E2E2E2", linewidth=0.45, which="both")
    panel_label(ax, "b")

    ax = axes[1, 0]
    case_order = catalog.case_id.tolist()
    method_order = list(PROFILE_STYLES)
    matrix = metrics.pivot(index="method", columns="case_id", values="primitive_geometric_mean_l2").loc[method_order, case_order].to_numpy(float)
    image = ax.imshow(np.log10(np.maximum(matrix, 1.0e-8)), aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(case_order))); ax.set_xticklabels(case_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(method_order))); ax.set_yticklabels([PROFILE_STYLES[m][1] for m in method_order], fontsize=5.5)
    ax.set_title(r"Per-case primary error, $\log_{10}$ scale")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10} E$")
    panel_label(ax, "c")

    ax = axes[1, 1]
    x = np.arange(len(catalog))
    ax.bar(x - 0.18, catalog.fixed_over_engineering_gain, width=0.36, color="#777777", label="fixed MC / CSE")
    ax.bar(x + 0.18, catalog.fno_over_engineering_gain, width=0.36, color=COLORS["FNO"], label="FNO / CSE")
    ax.axhline(1.0, color="#202020", ls="--", lw=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(catalog.case_id, rotation=35, ha="right")
    ax.set_ylabel("error gain over CSE-DKAN-FV")
    ax.set_title("Wins and same-grid failure controls")
    ax.legend(); ax.grid(axis="y", color="#E1E1E1", linewidth=0.45, which="both")
    panel_label(ax, "d")
    save_figure(fig, output, "fig8_training_and_case_error")


def method_label(name: str) -> str:
    return {
        "fno_operator": "FNO",
        "hllc_piecewise_constant": "128-cell HLLC, PC",
        "hllc_fixed_mc_subcell": "128-cell HLLC, MC",
        "physics_gated_dkan_fv_engineering": "CSE-DKAN-FV",
        "physics_gated_dkan_fv_sod_neighbourhood": "CSE-DKAN-FV (narrow)",
        "hllc_muscl_fine": "512-cell HLLC",
    }[name]


def method_color(name: str) -> str:
    return {
        "fno_operator": COLORS["FNO"],
        "hllc_piecewise_constant": "#BFC2C7",
        "hllc_fixed_mc_subcell": COLORS["Fixed MC"],
        "physics_gated_dkan_fv_engineering": COLORS["CSE-DKAN-FV"],
        "physics_gated_dkan_fv_sod_neighbourhood": "#C98786",
        "hllc_muscl_fine": COLORS["Fine HLLC"],
    }[name]


def figure_engineering(source: Path, output: Path) -> None:
    cases = pd.read_csv(source / "engineering_case_comparison.csv")
    methods = pd.read_csv(source / "engineering_method_group_summary.csv")
    methods = methods[methods.group == "in_distribution"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)

    ax = axes[0, 0]
    rng = np.random.default_rng(20260716)
    for position, group in enumerate(("in_distribution", "stress")):
        vals = cases[cases.case_group == group].fixed_over_engineering_gain.to_numpy(float)
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        color = "#6C86A3" if group == "in_distribution" else "#C77C67"
        ax.scatter(np.full(len(vals), position) + jitter, vals, s=16, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.9)
        ax.plot([position - 0.17, position + 0.17], [np.median(vals)] * 2, color="#202020", lw=1.6)
    ax.axhline(1.0, color="#707070", ls="--", lw=0.8)
    ax.axhline(1.05, color="#B64342", ls=":", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["In distribution\n(n=20 cases)", "Stress\n(n=10 cases)"])
    ax.set_ylabel("Fixed-MC error / CSE-DKAN-FV error")
    ax.set_title("Case-level paired gains")
    panel_label(ax, "a")

    ax = axes[0, 1]
    pareto_methods = {
        "fno_operator",
        "hllc_fixed_mc_subcell",
        "physics_gated_dkan_fv_engineering",
        "hllc_muscl_fine",
    }
    for _, row in methods[methods.method.isin(pareto_methods)].iterrows():
        label = method_label(row.method)
        ax.scatter(row.online_seconds_median, row.primary_error_geometric_mean, s=44 if row.method == "physics_gated_dkan_fv_engineering" else 32, color=method_color(row.method), edgecolor="white", linewidth=0.5, zorder=3)
        offset = {
            "fno_operator": (4, -10),
            "hllc_fixed_mc_subcell": (4, 7),
            "physics_gated_dkan_fv_engineering": (4, -10),
            "hllc_muscl_fine": (4, 4),
        }[row.method]
        ax.annotate(label, (row.online_seconds_median, row.primary_error_geometric_mean), xytext=offset, textcoords="offset points", fontsize=5.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median online time (s)")
    ax.set_ylabel("Geometric-mean state error")
    ax.grid(color="#E1E1E1", linewidth=0.45, which="both")
    ax.set_title("Accuracy-cost Pareto view")
    panel_label(ax, "b")

    ax = axes[1, 0]
    order = ["fno_operator", "hllc_fixed_mc_subcell", "physics_gated_dkan_fv_engineering", "hllc_muscl_fine"]
    subset = methods.set_index("method").loc[order]
    labels = [method_label(name) for name in order]
    values = subset.shock_width_median.to_numpy(float)
    ax.bar(np.arange(len(order)), values, color=[method_color(name) for name in order], edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Median 10-90% shock width")
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax.set_title("Shock resolution")
    panel_label(ax, "c")

    ax = axes[1, 1]
    x = np.arange(len(order))
    width = 0.36
    domain = subset.domain_integral_error_median.to_numpy(float)
    tv = subset.density_tv_excess_median.to_numpy(float)
    ax.bar(x - width / 2, domain, width, color="#668CB4", label="Domain-integral error")
    ax.bar(x + width / 2, tv, width, color="#D79B5B", label="Density-TV excess")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Relative indicator")
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax.legend(loc="upper left")
    ax.set_title("Conservation accuracy and oscillation audit")
    panel_label(ax, "d")
    save_figure(fig, output, "fig10_engineering_benchmark")


def figure_multiseed_ablation(source: Path, output: Path) -> None:
    seeds = pd.read_csv(source / "frozen_v3_seed_gains.csv")
    ablation = pd.read_csv(source / "ablation_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(seeds)) + 1
    ax.plot(x, seeds.burgers_gain, marker="o", ms=3.2, lw=1.0, color="#3E6D9C", label="Burgers")
    ax.plot(x, seeds.euler_gain, marker="s", ms=3.0, lw=1.0, color="#D18A47", label="Euler")
    ax.plot(x, seeds.cross_equation_gain, marker="D", ms=3.0, lw=1.1, color="#B64342", label="Cross-equation")
    ax.axhline(1.2, color="#777777", ls="--", lw=0.8, label="20% target")
    ax.set_xlabel("Matched seed index")
    ax.set_ylabel("Geometric error gain")
    ax.set_title("Frozen v3 multi-seed gains")
    ax.legend(loc="upper left")
    ax.grid(color="#E2E2E2", linewidth=0.45)
    panel_label(ax, "a")

    for ax, equation, label in ((axes[1], "burgers", "b"), (axes[2], "euler", "c")):
        subset = ablation[ablation.equation == equation].copy()
        names = [
            {
                "full": "Full",
                "no_discontinuity_basis": "No jump basis",
                "no_spatial_jump_condition": "No spatial gate",
                "no_viscous_transition_gate": "No viscosity gate",
                "no_tv_projection": "No TV projection",
                "no_trust_region": "No trust region",
                "no_resolution_fallback": "No resolution fallback",
            }[v]
            for v in subset.variant
        ]
        y = np.arange(len(names))
        colors = [COLORS["CSE-DKAN-FV"] if v == "Full" else "#AAB1BA" for v in names]
        ax.barh(y, subset.median_gain_vs_fixed.to_numpy(float), color=colors)
        ax.scatter(subset.minimum_pair_gain_vs_fixed.to_numpy(float), y, marker="|", s=65, color="#1D1D1D", label="Worst pair")
        ax.axvline(1.0, color="#777777", ls="--", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Gain over fixed reconstruction")
        ax.set_title(f"{equation.capitalize()} ablations")
        if equation == "burgers":
            ax.legend(loc="lower right")
        panel_label(ax, label)
    save_figure(fig, output, "fig11_multiseed_ablation")


def figure_tv_trust(source: Path, output: Path) -> None:
    data = pd.read_csv(source / "tv_trust_diagnostic.csv")
    data = data[data.method.str.contains("engineering") | (data.method == "hllc_fixed_mc_subcell")].copy()
    categories = ["Fixed MC", "0%", "2%", "5%", "10%", "No projection"]
    method_order = [
        "hllc_fixed_mc_subcell",
        "physics_gated_dkan_fv_engineering_tv_trust_0",
        "physics_gated_dkan_fv_engineering_tv_trust_0.02",
        "physics_gated_dkan_fv_engineering_tv_trust_0.05",
        "physics_gated_dkan_fv_engineering_tv_trust_0.1",
        "physics_gated_dkan_fv_engineering",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), constrained_layout=True)
    for group, color, marker in (("in_distribution", "#3E6D9C", "o"), ("stress", "#B64342", "s")):
        subset = data[data.group == group].set_index("method").loc[method_order]
        x = np.arange(len(categories))
        axes[0, 0].plot(x, subset.primary_error_geometric_mean, color=color, marker=marker, ms=4, lw=1.2, label=group.replace("_", " "))
        axes[0, 1].plot(x, 100 * subset.density_tv_excess_median, color=color, marker=marker, ms=4, lw=1.2)
        gain = subset.fixed_over_method_gain_median.to_numpy(float).copy()
        gain[0] = 1.0
        axes[1, 0].plot(x, gain, color=color, marker=marker, ms=4, lw=1.2)
        theta = subset.tv_trust_multiplier_median.to_numpy(float).copy()
        theta[0] = 0.0
        theta[-1] = 1.0
        axes[1, 1].plot(x, theta, color=color, marker=marker, ms=4, lw=1.2)
    titles = ["State error", "Density-TV excess", "Gain over fixed MC", "Accepted DKAN multiplier"]
    ylabels = ["Geometric-mean error", "TV excess (%)", "Paired median gain", "Median trust multiplier"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_title(titles[idx])
        ax.set_ylabel(ylabels[idx])
        ax.grid(color="#E1E1E1", linewidth=0.45)
        panel_label(ax, chr(ord("a") + idx))
    axes[0, 0].legend(loc="best")
    axes[1, 0].axhline(1.0, color="#777777", ls="--", lw=0.7)
    save_figure(fig, output, "fig12_tv_trust_diagnostic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_data)
    output = Path(args.output)
    # The full per-case atlases supersede the earlier compact casebook plates.
    for stem in (
        "fig4_casebook_in_a",
        "fig5_casebook_in_b",
        "fig6_casebook_stress_a",
        "fig7_casebook_stress_b",
        "fig9_canonical_baselines",
    ):
        for extension in ("svg", "pdf", "png", "tiff"):
            (output / f"{stem}.{extension}").unlink(missing_ok=True)
    figure_architecture(output)
    figure_dkan_mechanism(output)
    figure_constraint_logic(output)
    figure_training_and_case_error(source, output)
    figure_engineering(source, output)
    figure_multiseed_ablation(source, output)
    figure_tv_trust(source, output)
    diagnostic_names = []
    isolated_profile_names = []
    for number, case_id in enumerate(
        ["id01", "id08", "id11", "id18", "stress01", "stress04", "stress05", "stress10"],
        start=13,
    ):
        name = f"fig{number}_case_{case_id}"
        figure_case_diagnostic(source, output, case_id, name)
        diagnostic_names.append(name)
        isolated_name = f"fig{number + 8}_isolated_{case_id}"
        figure_model_isolated_profiles(source, output, case_id, isolated_name)
        isolated_profile_names.append(isolated_name)
    qa = {
        "backend": "Python/matplotlib only",
        "exports": ["SVG with editable text", "PDF with TrueType text", "PNG 300 dpi", "TIFF 600 dpi"],
        "source_data": sorted(path.name for path in source.glob("*.csv")),
        "figure_count": 23,
        "casebook_cases": ["id01", "id08", "id11", "id18", "stress01", "stress04", "stress05", "stress10"],
        "comparison_models_per_case": 6,
        "training_curve_note": "Training histories are family-level because the eight evaluation cases are not trained independently.",
        "per_case_loss_note": "Each case-specific loss curve is a time-resolved evaluation loss over physical rollout time, not a per-case training history.",
        "diagnostic_figures": diagnostic_names,
        "model_isolated_profile_figures": isolated_profile_names,
    }
    (output / "figure_qa_manifest.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": 23}, indent=2))


if __name__ == "__main__":
    main()
