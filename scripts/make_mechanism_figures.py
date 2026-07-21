"""Python-only publication figures for the mechanism-diverse Euler revision.

Figure contracts
----------------
Overview: the eight cases span distinct wave mechanisms, while the proposed
method is conservative and substantially stronger than FNO but only modestly
better than fixed MC on the same backbone.  Archetype: quantitative grid.

Case plates: every model receives a co-registered density prediction/error row;
profiles, time-resolved RMS error, the deterministic sensor gate, and component
errors explain where a visual difference changes the final metric.  Archetype:
asymmetric mixed-modality quantitative figure.

Sensor ablation: the explicit sensor removes the smooth-entropy regression and
improves Lax without changing checkpoints.  Archetype: quantitative grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


METHODS = [
    "fno_operator",
    "hllc_piecewise_constant",
    "hllc_fixed_mc_subcell",
    "physics_gated_dkan_fv_sod_neighbourhood",
    "sensor_gated_dkan_fv_engineering_tv_trust_0.02",
    "hllc_muscl_fine",
]
LABELS = {
    "fno_operator": "FNO",
    "hllc_piecewise_constant": "128-cell HLLC-PC",
    "hllc_fixed_mc_subcell": "128-cell HLLC-MC",
    "physics_gated_dkan_fv_sod_neighbourhood": "Narrow DKAN-FV",
    "sensor_gated_dkan_fv_engineering_tv_trust_0.02": "CSE-DKAN-FV",
    "hllc_muscl_fine": "512-cell HLLC",
}
COLORS = {
    "fno_operator": "#4A9D8F",
    "hllc_piecewise_constant": "#C7CBD1",
    "hllc_fixed_mc_subcell": "#7F8388",
    "physics_gated_dkan_fv_sod_neighbourhood": "#C98786",
    "sensor_gated_dkan_fv_engineering_tv_trust_0.02": "#B64342",
    "hllc_muscl_fine": "#294E75",
    "reference": "#151515",
}
LINESTYLES = {
    "fno_operator": "-",
    "hllc_piecewise_constant": ":",
    "hllc_fixed_mc_subcell": "--",
    "physics_gated_dkan_fv_sod_neighbourhood": "-.",
    "sensor_gated_dkan_fv_engineering_tv_trust_0.02": "-",
    "hllc_muscl_fine": "--",
}
CASE_TITLES = {
    "sod": "Sod three-wave tube",
    "lax": "Lax strong-shock tube",
    "moving_contact": "Isolated moving contact",
    "double_rarefaction": "Expansion-dominated double rarefaction",
    "colliding_streams": "Symmetric colliding streams",
    "shu_osher": "Shu–Osher shock–entropy interaction",
    "woodward_colella": "Woodward–Colella interacting blast",
    "smooth_entropy": "Smooth entropy-wave specificity control",
}
CASE_SHORT = {
    "sod": "Sod",
    "lax": "Lax",
    "moving_contact": "Contact",
    "double_rarefaction": "Rarefaction",
    "colliding_streams": "Collision",
    "shu_osher": "Shu–Osher",
    "woodward_colella": "Blast",
    "smooth_entropy": "Smooth wave",
}


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def figure_overview(summary: Path, trajectories: Path, output: Path) -> None:
    catalog = pd.read_csv(summary / "mechanism_case_catalog.csv")
    metrics = pd.read_csv(summary / "mechanism_metrics.csv")
    summary_json = json.loads((summary / "mechanism_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.35], hspace=0.55, wspace=0.42)
    for index, row in catalog.iterrows():
        ax = fig.add_subplot(grid[index // 4, index % 4])
        data = np.load(trajectories / f"{row.case_id}.npz")
        ax.plot(data["x"], data["initial"][:, 0], color="#65788C", lw=1.0, label="initial")
        ax.plot(data["x"], data["reference"][-1, :, 0], color="#151515", lw=1.25, label="reference")
        ax.set_title(CASE_SHORT[row.case_id], fontsize=6.4, pad=2)
        ax.set_xlim(0, 1)
        ax.grid(color="#E5E5E5", lw=0.35)
        if index % 4 == 0:
            ax.set_ylabel(r"Density $\rho$")
        if index // 4 == 1:
            ax.set_xlabel("x")
        panel_label(ax, chr(ord("a") + index), x=-0.18, y=1.10)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.50, 0.995))

    ax = fig.add_subplot(grid[2, :2])
    axes = [
        "single jump",
        "isolated contact",
        "expansion",
        "compression",
        "smooth oscillation",
        "multi-interface",
        "wall/periodic BC",
    ]
    matrix = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    ax.imshow(matrix, cmap=mpl.colors.ListedColormap(["#F2F3F4", "#476F91"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(axes)), axes, rotation=42, ha="right")
    ax.set_yticks(np.arange(len(catalog)), [CASE_SHORT[value] for value in catalog.case_id], fontsize=5.7)
    ax.tick_params(length=0)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x]:
                ax.text(x, y, "●", ha="center", va="center", color="white", fontsize=6)
    ax.set_title("Mechanism coverage fixed before evaluation")
    panel_label(ax, "i", x=-0.28)

    ax = fig.add_subplot(grid[2, 2:])
    gains = catalog.fixed_over_cse_gain.to_numpy(float)
    fno_gains = catalog.fno_over_cse_gain.to_numpy(float)
    positions = np.arange(len(catalog))
    ax.barh(positions + 0.17, fno_gains, height=0.30, color=COLORS["fno_operator"], label="FNO/CSE")
    ax.barh(positions - 0.17, gains, height=0.30, color=COLORS["sensor_gated_dkan_fv_engineering_tv_trust_0.02"], label="fixed MC/CSE")
    ax.axvline(1.0, color="#333333", lw=0.7)
    ax.set_xscale("log")
    ax.set_yticks(positions, [CASE_SHORT[value] for value in catalog.case_id], fontsize=5.7)
    ax.invert_yaxis()
    ax.set_xlabel("Error ratio (>1 favours CSE-DKAN-FV)")
    ax.grid(axis="x", which="both", color="#E0E0E0", lw=0.4)
    ax.legend(loc="lower right")
    ax.set_title(
        f"5/8 same-backbone wins; geometric gain {summary_json['same_backbone_geometric_mean_gain']:.3f}"
    )
    panel_label(ax, "j", x=-0.22)
    fig.suptitle(
        "A mechanism-diverse, frozen Euler generalization suite",
        y=1.018,
        fontsize=9,
        fontweight="bold",
    )
    save_figure(fig, output, "fig12_mechanism_suite_overview")


def case_plate(case_id: str, summary: Path, trajectories: Path, output: Path, number: int) -> None:
    data = np.load(trajectories / f"{case_id}.npz")
    metrics = pd.read_csv(summary / "mechanism_metrics.csv")
    losses = pd.read_csv(summary / "mechanism_time_losses.csv")
    catalog = pd.read_csv(summary / "mechanism_case_catalog.csv").set_index("case_id")
    case_metrics = metrics[metrics.case_id == case_id].set_index("method")
    case_losses = losses[losses.case_id == case_id]
    x = data["x"]
    times = data["times"]
    reference = data["reference"]
    physical_methods = [method for method in METHODS if method != "fno_operator"]
    density_values = np.concatenate([reference.ravel()] + [data[m][..., 0].ravel() for m in physical_methods])
    lower, upper = np.quantile(density_values, [0.002, 0.998])
    span = max(upper - lower, 1.0e-8)
    density_min, density_max = lower - 0.04 * span, upper + 0.04 * span
    physical_errors = np.concatenate(
        [np.abs(data[m][..., 0] - reference[..., 0]).ravel() for m in physical_methods]
    )
    error_max = max(float(np.quantile(physical_errors, 0.997)), 1.0e-8)
    error_min = max(error_max * 1.0e-4, 1.0e-9)

    fig = plt.figure(figsize=(7.2, 8.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.65, 2.15], hspace=0.20, left=0.10, right=0.945, bottom=0.07, top=0.89)
    atlas = outer[0].subgridspec(6, 4, width_ratios=[1.0, 0.030, 1.0, 0.030], hspace=0.055, wspace=0.13)
    result_axes, error_axes = [], []
    result_image = error_image = None
    contour_levels = np.linspace(float(reference[..., 0].min()), float(reference[..., 0].max()), 7)[1:-1]
    for row, method in enumerate(METHODS):
        result_ax = fig.add_subplot(atlas[row, 0])
        error_ax = fig.add_subplot(atlas[row, 2])
        result_axes.append(result_ax)
        error_axes.append(error_ax)
        result_image = result_ax.pcolormesh(
            x, times, np.clip(data[method][..., 0], density_min, density_max),
            cmap="cividis", shading="auto", vmin=density_min, vmax=density_max, rasterized=True,
        )
        if contour_levels.size:
            result_ax.contour(x, times, reference[..., 0], levels=contour_levels, colors="#111111", linewidths=0.28, alpha=0.65)
        absolute_error = np.abs(data[method][..., 0] - reference[..., 0])
        error_image = error_ax.pcolormesh(
            x, times, np.maximum(np.clip(absolute_error, None, error_max), error_min),
            cmap="magma", shading="auto", norm=LogNorm(vmin=error_min, vmax=error_max), rasterized=True,
        )
        result_ax.set_ylabel(LABELS[method], color=COLORS[method], fontsize=5.2, labelpad=3)
        result_ax.tick_params(labelsize=5, length=2)
        error_ax.tick_params(labelsize=5, length=2)
        error_ax.set_yticklabels([])
        if row < 5:
            result_ax.set_xticklabels([])
            error_ax.set_xticklabels([])
        else:
            result_ax.set_xlabel("x")
            error_ax.set_xlabel("x")
    result_axes[0].set_title("Density prediction (reference contours)", fontsize=6.8)
    error_axes[0].set_title("Absolute density error (shared logarithmic scale)", fontsize=6.8)
    result_axes[2].set_ylabel("128-cell HLLC-MC", color=COLORS["hllc_fixed_mc_subcell"], fontsize=5.2, labelpad=3)
    fig.colorbar(result_image, cax=fig.add_subplot(atlas[:, 1]), label=r"$\rho$")
    fig.colorbar(error_image, cax=fig.add_subplot(atlas[:, 3]), label=r"$|\hat\rho-\rho_{ref}|$")
    result_axes[0].text(-0.24, 1.23, "a", transform=result_axes[0].transAxes, fontsize=8, fontweight="bold")

    support = outer[1].subgridspec(2, 4, hspace=0.62, wspace=0.78)
    variables = ((0, r"Density $\rho$"), (1, r"Velocity $u$"), (2, r"Pressure $p$"))
    final_reference = reference[-1]
    for column, (component, label) in enumerate(variables):
        ax = fig.add_subplot(support[0, column])
        ax.plot(x, final_reference[:, component], color=COLORS["reference"], lw=2.6, alpha=0.35, label="Reference")
        all_physical = [final_reference[:, component]] + [data[m][-1, :, component] for m in physical_methods]
        values = np.concatenate(all_physical)
        low, high = np.quantile(values, [0.002, 0.998])
        margin = max(0.08 * (high - low), 1.0e-4)
        for method in METHODS:
            ax.plot(
                x, data[method][-1, :, component], color=COLORS[method], ls=LINESTYLES[method],
                lw=1.25 if method == "sensor_gated_dkan_fv_engineering_tv_trust_0.02" else 0.8,
                label=LABELS[method], clip_on=True,
            )
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(color="#E4E4E4", lw=0.35)
        panel_label(ax, chr(ord("b") + column))

    ax = fig.add_subplot(support[0, 3])
    for method in METHODS:
        block = case_losses[case_losses.method == method]
        ax.plot(block.time_fraction, block.primitive_rms_l2, color=COLORS[method], ls=LINESTYLES[method], lw=1.2 if "sensor_gated" in method else 0.75, label=LABELS[method])
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized time $t/t_f$")
    ax.set_ylabel("Primitive RMS error")
    ax.grid(color="#E4E4E4", lw=0.35, which="both")
    panel_label(ax, "e")

    ax = fig.add_subplot(support[1, 0])
    extent = [0, 1, float(times[0]), float(times[-1])]
    sensor = np.repeat(data["cse_blend"], x.size // data["cse_blend"].shape[1], axis=1)
    image = ax.imshow(sensor, origin="lower", aspect="auto", extent=extent, cmap="Blues", vmin=0, vmax=max(float(np.max(sensor)), 0.15))
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title("Effective sensor × DKAN gate", fontsize=6.6)
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    panel_label(ax, "f", x=-0.18, y=1.10)

    ax = fig.add_subplot(support[1, 1:3])
    component_matrix = case_metrics.loc[METHODS, ["density_normalized_l2", "velocity_normalized_l2", "pressure_normalized_l2"]].to_numpy(float)
    image = ax.imshow(np.log10(np.maximum(component_matrix, 1.0e-12)), aspect="auto", cmap="magma_r")
    ax.set_xticks([0, 1, 2], [r"$E_\rho$", r"$E_u$", r"$E_p$"])
    heatmap_labels = ["FNO", "HLLC-PC", "HLLC-MC", "DKAN narrow", "CSE-DKAN-FV", "HLLC-512"]
    ax.set_yticks(np.arange(len(METHODS)), heatmap_labels, fontsize=5.1)
    for y in range(component_matrix.shape[0]):
        for xx in range(component_matrix.shape[1]):
            value = component_matrix[y, xx]
            ax.text(xx, y, f"{value:.2g}", ha="center", va="center", fontsize=4.7, color="white" if np.log10(max(value, 1e-12)) < np.nanmedian(np.log10(np.maximum(component_matrix, 1e-12))) else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label=r"$\log_{10} E$")
    ax.set_title("Final primitive-component errors", fontsize=6.6)
    panel_label(ax, "g", x=-0.18)

    ax = fig.add_subplot(support[1, 3])
    ax.axis("off")
    fixed = case_metrics.loc["hllc_fixed_mc_subcell"]
    cse = case_metrics.loc["sensor_gated_dkan_fv_engineering_tv_trust_0.02"]
    info = catalog.loc[case_id]
    text = (
        f"fixed MC / CSE: {info.fixed_over_cse_gain:.3f}×\n\n"
        f"FNO / CSE: {info.fno_over_cse_gain:.2f}×\n\n"
        f"min $\\rho$, min $p$:\n{cse.minimum_density:.3g}, {cse.minimum_pressure:.3g}\n\n"
        f"TV excess: {100*cse.density_tv_excess:.2f}%"
    )
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#F7F8FA", edgecolor="#B9C0C8", lw=0.8))
    ax.text(0.07, 0.90, text, va="top", fontsize=5.5, linespacing=1.08)
    panel_label(ax, "h", x=-0.10)

    handles, labels = result_axes[0].get_legend_handles_labels()
    profile_handles, profile_labels = fig.axes[-8].get_legend_handles_labels()
    fig.legend(profile_handles, profile_labels, loc="upper center", ncol=7, bbox_to_anchor=(0.52, 0.936), fontsize=5.3)
    fig.suptitle(
        f"{CASE_TITLES[case_id]}: six-model result and error audit",
        y=0.985,
        fontsize=8.5,
        fontweight="bold",
    )
    save_figure(fig, output, f"fig{number}_mechanism_{case_id}")


def figure_sensor_ablation(summary: Path, output: Path) -> None:
    data = pd.read_csv(summary / "sensor_ablation.csv")
    catalog = pd.read_csv(summary / "mechanism_case_catalog.csv").set_index("case_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax = axes[0]
    positions = np.arange(len(data))
    width = 0.36
    ax.bar(positions - width / 2, data.without_sensor, width, color="#C98786", label="without explicit sensor")
    ax.bar(positions + width / 2, data.with_sensor, width, color=COLORS["sensor_gated_dkan_fv_engineering_tv_trust_0.02"], label="with sensor")
    ax.set_yscale("log")
    ax.set_xticks(positions, data.case_id, rotation=35, ha="right")
    ax.set_ylabel("Final primitive RMS error")
    ax.grid(axis="y", color="#E1E1E1", lw=0.4, which="both")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    gains = data.gain.to_numpy(float)
    colors = np.where(gains >= 1.0, "#4A9D8F", "#B9BDC6")
    ax.barh(np.arange(len(data)), gains, color=colors)
    ax.axvline(1.0, color="#333333", lw=0.7)
    ax.set_yticks(np.arange(len(data)), data.case_id)
    ax.invert_yaxis()
    ax.set_xlabel("without-sensor / sensor error")
    ax.set_title("Smooth specificity is recovered without retraining")
    ax.grid(axis="x", color="#E1E1E1", lw=0.4)
    panel_label(ax, "b", x=-0.22)
    save_figure(fig, output, "fig21_sensor_ablation")


def figure_reference_convergence(summary: Path, output: Path) -> None:
    data = pd.read_csv(summary / "mechanism_reference_convergence.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    variables = ["density_normalized_l2", "velocity_normalized_l2", "pressure_normalized_l2", "primitive_rms_l2"]
    labels = [r"$E_\rho$", r"$E_u$", r"$E_p$", "RMS"]
    x = np.arange(len(data))
    width = 0.18
    colors = ["#557A9F", "#8FA4BF", "#C98786", "#B64342"]
    for index, (variable, label, color) in enumerate(zip(variables, labels, colors)):
        ax.bar(x + (index - 1.5) * width, data[variable], width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, ["Shu–Osher\n2048 vs 4096", "Blast\n2048 vs 4096"])
    ax.set_ylabel("Reference-grid disagreement")
    ax.grid(axis="y", which="both", color="#E1E1E1", lw=0.4)
    ax.legend(ncol=2)
    panel_label(ax, "a", x=-0.18)
    save_figure(fig, output, "fig22_reference_convergence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = Path(args.summary)
    trajectories = Path(args.trajectories)
    output = Path(args.output)
    figure_overview(summary, trajectories, output)
    for number, case_id in enumerate(CASE_TITLES, start=13):
        case_plate(case_id, summary, trajectories, output, number)
    figure_sensor_ablation(summary, output)
    figure_reference_convergence(summary, output)
    print(json.dumps({"figures": 11, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
