"""Export manuscript source data and representative Euler profiles.

This script performs no plotting. It converts frozen JSON outputs and selected
checkpoints into auditable CSV files consumed by the Python-only figure script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cse_dkan.euler_finite_volume import conservative_to_primitive, primitive_to_conservative
from cse_dkan.euler_subcell import (
    ConservativeEulerDKANSubcell,
    density_tv_trust_projection,
)
from cse_dkan.metrics import normalized_lp
from cse_dkan.reference import ExactSodSolver, PrimitiveState
from run_engineering_riemann_benchmark import (
    cached_fv_state,
    fno_input,
    load_fno,
    load_hybrid,
)


def metric_median(block: dict, key: str) -> float:
    return float(block[key]["median"])


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_canonical(root: Path, output: Path) -> None:
    pinn = json.loads(
        (root / "results/canonical_pinn_baselines_v1/summary.json").read_text()
    )
    dkan = json.loads(
        (root / "results/canonical_dkan_pinn_extension_v1/summary.json").read_text()
    )
    labels = {
        "mlp_pinn": "MLP-PINN",
        "gw_pinn": "GW-PINN",
        "av_pinn": "AV-PINN",
        "ci_pinn": "CI-PINN",
    }
    rows = []
    for equation in ("burgers", "euler"):
        error_key = "normalized_l2" if equation == "burgers" else "primitive_geometric_mean_l2"
        for key, label in labels.items():
            block = pinn[equation][key]
            conservation_key = (
                "max_mass_drift" if equation == "burgers" else "max_energy_balance_error"
            )
            rows.append(
                {
                    "equation": equation,
                    "method": label,
                    "state_error": metric_median(block, error_key),
                    "shock_width": metric_median(block, "shock_width_10_90"),
                    "conservation_indicator": metric_median(block, conservation_key),
                    "entropy_violation": metric_median(block, "entropy_violation_max" if equation == "burgers" else "shock_entropy_violation"),
                    "training_seconds": metric_median(block, "training_seconds"),
                    "online_seconds": metric_median(block, "inference_seconds"),
                    "seed_count": pinn["seed_count"],
                }
            )
        dblock = dkan[equation]["metrics"]
        rows.append(
            {
                "equation": equation,
                "method": "DKAN-PINN",
                "state_error": metric_median(dblock, error_key),
                "shock_width": metric_median(dblock, "shock_width_10_90"),
                "conservation_indicator": metric_median(
                    dblock, "max_mass_drift" if equation == "burgers" else "max_energy_balance_error"
                ),
                "entropy_violation": metric_median(
                    dblock, "entropy_violation_max" if equation == "burgers" else "shock_entropy_violation"
                ),
                "training_seconds": metric_median(dblock, "training_seconds"),
                "online_seconds": metric_median(dblock, "inference_seconds"),
                "seed_count": len(dkan[equation]["raw"]),
            }
        )
        context = pinn["hybrid_context"][equation]
        rows.extend(
            [
                {
                    "equation": equation,
                    "method": "Fixed FV reconstruction",
                    "state_error": metric_median(context, "fixed_l2"),
                    "shock_width": "",
                    "conservation_indicator": 0.0,
                    "entropy_violation": 0.0,
                    "training_seconds": 0.0,
                    "online_seconds": metric_median(context, "fv_seconds"),
                    "seed_count": 5,
                },
                {
                    "equation": equation,
                    "method": "CSE-DKAN-FV",
                    "state_error": metric_median(context, "hybrid_l2"),
                    "shock_width": metric_median(context, "shock_width_10_90"),
                    "conservation_indicator": 4.44e-16 if equation == "euler" else 2.22e-16,
                    "entropy_violation": 0.0,
                    "training_seconds": metric_median(context, "pretraining_seconds"),
                    "online_seconds": metric_median(context, "fv_seconds") + metric_median(context, "reconstruction_seconds"),
                    "seed_count": 5,
                },
            ]
        )
    write_rows(output / "canonical_method_comparison.csv", rows)


def export_frozen_summaries(root: Path, output: Path) -> None:
    gate = json.loads((root / "results/final_float64_v3/gate_summary.json").read_text())
    seed_rows = []
    for index, value in enumerate(gate["cross_equation"]["per_seed_gains"]):
        seed_rows.append(
            {
                "seed_index": index + 1,
                "burgers_gain": gate["burgers"]["seed_aggregates"][index]["geometric_mean_gain"],
                "euler_gain": gate["euler"]["seed_aggregates"][index]["geometric_mean_gain"],
                "cross_equation_gain": value,
            }
        )
    write_rows(output / "frozen_v3_seed_gains.csv", seed_rows)

    ablations = json.loads(
        (root / "results/final_hybrid_ablations_v1/summary.json").read_text()
    )
    rows = []
    for equation in ("burgers", "euler"):
        for name, block in ablations[equation].items():
            seed_rows = block["seed_rows"]
            def extreme(key: str, reducer, default=""):
                values = [row[key] for row in seed_rows if key in row]
                return reducer(values) if values else default
            rows.append(
                {
                    "equation": equation,
                    "variant": name,
                    "median_gain_vs_fixed": block["median_gain_vs_fixed"],
                    "minimum_pair_gain_vs_fixed": block["minimum_pair_gain_vs_fixed"],
                    "maximum_pair_error_ratio_vs_full": block["maximum_pair_error_ratio_vs_full"],
                    "maximum_tv_excess": extreme("maximum_tv_excess", max),
                    "maximum_rh_ratio": extreme("maximum_rh_ratio", max),
                    "minimum_density": extreme("minimum_density", min),
                    "minimum_pressure": extreme("minimum_pressure", min),
                    "maximum_entropy_violation": extreme("maximum_entropy_violation", max),
                }
            )
    write_rows(output / "ablation_summary.csv", rows)

    for source, target in (
        (
            root / "results/engineering_riemann_v1/summary/method_group_summary.csv",
            output / "engineering_method_group_summary.csv",
        ),
        (
            root / "results/engineering_riemann_v1/summary/gate_results.csv",
            output / "engineering_gate_results.csv",
        ),
        (
            root / "results/engineering_riemann_v1/summary/case_comparison.csv",
            output / "engineering_case_comparison.csv",
        ),
        (
            root / "results/engineering_riemann_v1/summary/tv_trust_diagnostic.csv",
            output / "tv_trust_diagnostic.csv",
        ),
    ):
        target.write_bytes(source.read_bytes())


CASEBOOK_IDS = [
    "id01",
    "id08",
    "id11",
    "id18",
    "stress01",
    "stress04",
    "stress05",
    "stress10",
]


CASEBOOK_DESCRIPTIONS = {
    "id01": "Representative in-range pressure-driven tube with moderate density, velocity, and pressure contrasts.",
    "id08": "High-left-pressure case with a weak density contrast; retained because fixed MC is slightly more accurate.",
    "id11": "Long-time low-right-pressure case near the lower pressure edge of the training range.",
    "id18": "Strong pressure jump with co-directed negative velocities; the largest same-grid gain among the selected in-range cases.",
    "stress01": "Blast-like extrapolation with a 500:1 pressure ratio and a 10:1 density ratio.",
    "stress04": "Symmetric colliding streams that generate a compression-dominated central interaction.",
    "stress05": "Reversed density and pressure step; retained as an out-of-distribution same-grid failure case.",
    "stress10": "Classical Sod shock tube evaluated later than the engineering training times; retained as a familiar failure control.",
}


def export_training_histories(root: Path, output: Path) -> None:
    rows = []
    specifications = []
    for seed in (71001, 71002, 71003):
        specifications.append(
            (
                "FNO",
                seed,
                root / f"results/engineering_riemann_v1/fno/seed_{seed}.json",
                "training_weighted_mse",
                "validation_mse",
                None,
            )
        )
    for label, seeds, template in (
        (
            "CSE-DKAN-FV",
            (72001, 72002, 72003),
            "results/engineering_riemann_v1/hybrid/seed_{seed}/model.json",
        ),
        (
            "CSE-DKAN-FV (narrow)",
            (26001, 26002, 26003),
            "results/final_float64_v1/euler_models/seed_{seed}/model.json",
        ),
    ):
        for seed in seeds:
            specifications.append(
                (
                    label,
                    seed,
                    root / template.format(seed=seed),
                    "training_oracle_mse",
                    "validation_oracle_mse",
                    "validation_profile_mse",
                )
            )
    for method, seed, path, train_key, validation_key, profile_key in specifications:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["history"]:
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "iteration": item["iteration"],
                    "training_loss": item[train_key],
                    "validation_loss": item[validation_key],
                    "profile_validation_loss": (
                        item[profile_key] if profile_key is not None else ""
                    ),
                }
            )
    write_rows(output / "training_histories.csv", rows)


def export_casebook_metrics(root: Path, output: Path, case_ids: list[str]) -> None:
    payload = json.loads(
        (root / "results/engineering_riemann_v1/benchmark_raw.json").read_text(
            encoding="utf-8"
        )
    )
    method_order = [
        "fno_operator",
        "hllc_piecewise_constant",
        "hllc_fixed_mc_subcell",
        "physics_gated_dkan_fv_sod_neighbourhood",
        "physics_gated_dkan_fv_engineering",
        "hllc_muscl_fine",
    ]
    selected = [row for row in payload["rows"] if row["case_id"] in case_ids]
    metric_keys = [
        "density_normalized_l2",
        "velocity_normalized_l2",
        "pressure_normalized_l2",
        "primitive_geometric_mean_l2",
        "maximum_relative_conservation_error",
        "minimum_density",
        "minimum_pressure",
        "density_tv_excess",
        "online_seconds",
        "maximum_cell_average_change",
        "mean_shock_width_10_90",
        "maximum_shock_entropy_violation",
        "maximum_normalized_rh_residual",
    ]
    rows = []
    for case_id in case_ids:
        for method in method_order:
            block = [
                row
                for row in selected
                if row["case_id"] == case_id and row["method"] == method
            ]
            if not block:
                raise RuntimeError(f"missing casebook rows for {case_id}/{method}")
            result = {
                "case_id": case_id,
                "case_group": block[0]["case_group"],
                "method": method,
                "seed_count": len(block),
            }
            for key in metric_keys:
                values = [
                    float(row[key])
                    for row in block
                    if row.get(key) is not None
                ]
                result[key] = float(np.median(values)) if values else ""
            rows.append(result)
    write_rows(output / "casebook_metrics.csv", rows)

    protocol = json.loads(
        (root / "protocols/engineering_riemann_operator_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cases = {case["id"]: case for case in protocol["cases"]}
    comparisons = {}
    comparison_path = root / "results/engineering_riemann_v1/summary/case_comparison.csv"
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            comparisons[row["case_id"]] = row
    catalog = []
    for case_id in case_ids:
        case = cases[case_id]
        comparison = comparisons[case_id]
        catalog.append(
            {
                "case_id": case_id,
                "case_group": case["group"],
                "left_density": case["left"][0],
                "left_velocity": case["left"][1],
                "left_pressure": case["left"][2],
                "right_density": case["right"][0],
                "right_velocity": case["right"][1],
                "right_pressure": case["right"][2],
                "final_time": case["time"],
                "fixed_over_engineering_gain": comparison[
                    "fixed_over_engineering_gain"
                ],
                "fno_over_engineering_gain": comparison[
                    "fno_over_engineering_gain"
                ],
                "description": CASEBOOK_DESCRIPTIONS[case_id],
            }
        )
    write_rows(output / "casebook_catalog.csv", catalog)


def profile_quantiles(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack(values, axis=0)
    return (
        np.median(stacked, axis=0),
        np.min(stacked, axis=0),
        np.max(stacked, axis=0),
    )


def component_and_primary_losses(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[list[float], float]:
    """Return the three normalized L2 components and their geometric mean."""
    components = [
        normalized_lp(prediction[:, component], reference[:, component], 2)
        for component in range(3)
    ]
    primary = float(
        np.exp(np.mean(np.log(np.maximum(np.asarray(components), 1.0e-15))))
    )
    return components, primary


def export_case_diagnostics(
    root: Path,
    output: Path,
    case_ids: list[str],
    fno_checkpoints: list[Path],
    legacy_checkpoints: list[Path],
    hybrid_checkpoints: list[Path],
    time_samples: int = 13,
) -> None:
    """Export time-resolved, per-case diagnostics without inventing training runs.

    The learned models were trained once on families of Riemann problems.  The
    exported ``case_loss`` files therefore contain an explicitly named
    evaluation loss over physical rollout time, not a per-case optimization
    history.  This distinction is carried into every figure caption.
    """
    protocol = json.loads(
        (root / "protocols/engineering_riemann_operator_v1.json").read_text()
    )
    configuration = protocol["benchmark"]
    points = int(configuration["evaluation_points"])
    coarse_cells = int(configuration["coarse_cells"])
    fine_cells = int(configuration["fine_hllc_cells"])
    refinement = points // coarse_cells
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_model = ConservativeEulerDKANSubcell(hidden_width=16, correction_scale=0.0).to(
        device=device, dtype=torch.float64
    ).eval()
    coordinates = fixed_model.uniform_coordinates(
        refinement, dtype=torch.float64, device=device
    )
    fno_models = [load_fno(path, device) for path in fno_checkpoints]
    legacy_models = [load_hybrid(path, device) for path in legacy_checkpoints]
    hybrid_models = [load_hybrid(path, device) for path in hybrid_checkpoints]
    x = (np.arange(points, dtype=np.float64) + 0.5) / points
    coarse_x = (np.arange(coarse_cells, dtype=np.float64) + 0.5) / coarse_cells
    fine_x = (np.arange(fine_cells, dtype=np.float64) + 0.5) / fine_cells
    cache = root / "results/engineering_riemann_v1/state_cache"
    cases = {case["id"]: case for case in protocol["cases"]}
    method_order = [
        "fno_operator",
        "hllc_piecewise_constant",
        "hllc_fixed_mc_subcell",
        "physics_gated_dkan_fv_sod_neighbourhood",
        "physics_gated_dkan_fv_engineering",
        "hllc_muscl_fine",
    ]

    for case_id in case_ids:
        case = cases[case_id]
        final_time = float(case["time"])
        exact = ExactSodSolver(
            left=PrimitiveState(*case["left"]), right=PrimitiveState(*case["right"])
        )
        coarse_primitive = np.where(
            (coarse_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        coarse_initial = primitive_to_conservative(
            torch.tensor(coarse_primitive, dtype=torch.float64, device=device)
        )
        fine_primitive = np.where(
            (fine_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        fine_initial = primitive_to_conservative(
            torch.tensor(fine_primitive, dtype=torch.float64, device=device)
        )

        trajectory_rows: list[dict] = []
        loss_rows: list[dict] = []
        for time_index, physical_time in enumerate(
            np.linspace(0.0, final_time, time_samples)
        ):
            physical_time = float(physical_time)
            exact_profile = np.column_stack(exact.sample(x, physical_time))
            if time_index == 0:
                coarse_state = coarse_initial
                fine_state = fine_initial
            else:
                coarse_state, _ = cached_fv_state(
                    coarse_initial, physical_time, coarse_cells, cache
                )
                fine_state, _ = cached_fv_state(
                    fine_initial, physical_time, fine_cells, cache
                )

            fine_profile = conservative_to_primitive(fine_state).detach().cpu().numpy()
            if fine_cells != points:
                fine_profile = np.column_stack(
                    [np.interp(x, fine_x, fine_profile[:, c]) for c in range(3)]
                )
            with torch.no_grad():
                fixed_conservative = fixed_model(coarse_state, coordinates)
            fixed_profile = (
                conservative_to_primitive(fixed_conservative)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )
            piecewise_profile = (
                conservative_to_primitive(
                    coarse_state.unsqueeze(-2).expand(-1, refinement, -1)
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )

            hybrid_profiles, hybrid_blends = [], []
            for model, _, _, _ in hybrid_models:
                with torch.no_grad():
                    candidate, blend = model(
                        coarse_state,
                        coordinates,
                        learned_blend_multiplier=configuration[
                            "hybrid_blend_multiplier"
                        ],
                        return_blend=True,
                    )
                hybrid_profiles.append(
                    conservative_to_primitive(candidate)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)
                )
                hybrid_blends.append(
                    np.repeat(blend.detach().cpu().numpy(), refinement)
                )
            hybrid_profile = np.median(np.stack(hybrid_profiles), axis=0)
            hybrid_blend = np.median(np.stack(hybrid_blends), axis=0)

            legacy_profiles = []
            for model, _, _, _ in legacy_models:
                with torch.no_grad():
                    candidate = model(
                        coarse_state,
                        coordinates,
                        learned_blend_multiplier=configuration[
                            "hybrid_blend_multiplier"
                        ],
                    )
                legacy_profiles.append(
                    conservative_to_primitive(candidate)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)
                )
            legacy_profile = np.median(np.stack(legacy_profiles), axis=0)

            fno_profiles = []
            for model, normalization, _, _ in fno_models:
                values = fno_input(case, points, physical_time, device)
                normalized = (
                    values - normalization["input_mean"]
                ) / normalization["input_std"]
                with torch.no_grad():
                    transformed = (
                        model(normalized) * normalization["target_std"]
                        + normalization["target_mean"]
                    )[0].cpu().numpy()
                fno_profiles.append(
                    np.column_stack(
                        (
                            np.exp(transformed[:, 0]),
                            transformed[:, 1],
                            np.exp(transformed[:, 2]),
                        )
                    )
                )
            fno_profile = np.median(np.stack(fno_profiles), axis=0)

            profiles = {
                "fno_operator": fno_profile,
                "hllc_piecewise_constant": piecewise_profile,
                "hllc_fixed_mc_subcell": fixed_profile,
                "physics_gated_dkan_fv_sod_neighbourhood": legacy_profile,
                "physics_gated_dkan_fv_engineering": hybrid_profile,
                "hllc_muscl_fine": fine_profile,
            }
            if time_index > 0:
                for method in method_order:
                    components, primary = component_and_primary_losses(
                        profiles[method], exact_profile
                    )
                    loss_rows.append(
                        {
                            "case_id": case_id,
                            "time": physical_time,
                            "time_fraction": physical_time / final_time,
                            "method": method,
                            "density_loss": components[0],
                            "velocity_loss": components[1],
                            "pressure_loss": components[2],
                            "case_evaluation_loss": primary,
                        }
                    )

            for index, coordinate in enumerate(x):
                row = {
                    "case_id": case_id,
                    "time": physical_time,
                    "time_fraction": (
                        physical_time / final_time if final_time > 0.0 else 0.0
                    ),
                    "x": coordinate,
                    "exact_density": exact_profile[index, 0],
                    # Retain the original CSE columns for backward-compatible audits.
                    "cse_density": hybrid_profile[index, 0],
                    "absolute_density_error": abs(
                        hybrid_profile[index, 0] - exact_profile[index, 0]
                    ),
                    "cse_blend": hybrid_blend[index],
                }
                # A co-registered density result and error field for every model
                # allows the manuscript to compare local wave errors directly,
                # rather than inferring them from a scalar final-time score.
                for method in method_order:
                    density = profiles[method][index, 0]
                    row[f"{method}_density"] = density
                    row[f"{method}_absolute_density_error"] = abs(
                        density - exact_profile[index, 0]
                    )
                trajectory_rows.append(row)
        write_rows(output / f"trajectory_{case_id}.csv", trajectory_rows)
        write_rows(output / f"case_loss_{case_id}.csv", loss_rows)


def export_profiles(
    root: Path,
    output: Path,
    case_ids: list[str],
    fno_checkpoints: list[Path],
    legacy_checkpoints: list[Path],
    hybrid_checkpoints: list[Path],
) -> None:
    protocol = json.loads(
        (root / "protocols/engineering_riemann_operator_v1.json").read_text()
    )
    configuration = protocol["benchmark"]
    points = int(configuration["evaluation_points"])
    coarse_cells = int(configuration["coarse_cells"])
    fine_cells = int(configuration["fine_hllc_cells"])
    refinement = points // coarse_cells
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_model = ConservativeEulerDKANSubcell(hidden_width=16, correction_scale=0.0).to(
        device=device, dtype=torch.float64
    ).eval()
    coordinates = fixed_model.uniform_coordinates(
        refinement, dtype=torch.float64, device=device
    )
    fno_models = [load_fno(path, device) for path in fno_checkpoints]
    legacy_models = [load_hybrid(path, device) for path in legacy_checkpoints]
    hybrid_models = [load_hybrid(path, device) for path in hybrid_checkpoints]
    x = (np.arange(points, dtype=np.float64) + 0.5) / points
    cache = root / "results/engineering_riemann_v1/state_cache"

    cases = {case["id"]: case for case in protocol["cases"]}
    for case_id in case_ids:
        case = cases[case_id]
        final_time = float(case["time"])
        exact = ExactSodSolver(
            left=PrimitiveState(*case["left"]), right=PrimitiveState(*case["right"])
        )
        exact_profile = np.column_stack(exact.sample(x, final_time))
        coarse_x = (np.arange(coarse_cells) + 0.5) / coarse_cells
        coarse_primitive = np.where(
            (coarse_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        coarse_initial = primitive_to_conservative(
            torch.tensor(coarse_primitive, dtype=torch.float64, device=device)
        )
        coarse_state, _ = cached_fv_state(
            coarse_initial, final_time, coarse_cells, cache
        )
        fine_x = (np.arange(fine_cells) + 0.5) / fine_cells
        fine_primitive = np.where(
            (fine_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        fine_initial = primitive_to_conservative(
            torch.tensor(fine_primitive, dtype=torch.float64, device=device)
        )
        fine_state, _ = cached_fv_state(fine_initial, final_time, fine_cells, cache)
        fine_profile = conservative_to_primitive(fine_state).detach().cpu().numpy()
        if fine_cells != points:
            fine_profile = np.column_stack(
                [np.interp(x, fine_x, fine_profile[:, c]) for c in range(3)]
            )
        with torch.no_grad():
            fixed_conservative = fixed_model(coarse_state, coordinates)
        fixed_profile = (
            conservative_to_primitive(fixed_conservative)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)
        )
        piecewise_profile = (
            conservative_to_primitive(
                coarse_state.unsqueeze(-2).expand(-1, refinement, -1)
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)
        )

        hybrid_profiles, trusted_profiles = [], []
        for model, _, _, _ in hybrid_models:
            with torch.no_grad():
                candidate = model(
                    coarse_state,
                    coordinates,
                    learned_blend_multiplier=configuration["hybrid_blend_multiplier"],
                )
                trusted, _ = density_tv_trust_projection(
                    fixed_conservative, candidate, relative_budget=0.02
                )
            hybrid_profiles.append(
                conservative_to_primitive(candidate)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )
            trusted_profiles.append(
                conservative_to_primitive(trusted)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )

        legacy_profiles = []
        for model, _, _, _ in legacy_models:
            with torch.no_grad():
                candidate = model(
                    coarse_state,
                    coordinates,
                    learned_blend_multiplier=configuration[
                        "hybrid_blend_multiplier"
                    ],
                )
            legacy_profiles.append(
                conservative_to_primitive(candidate)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)
            )

        fno_profiles = []
        for model, normalization, _, _ in fno_models:
            values = fno_input(case, points, final_time, device)
            normalized = (values - normalization["input_mean"]) / normalization["input_std"]
            with torch.no_grad():
                transformed = (
                    model(normalized) * normalization["target_std"]
                    + normalization["target_mean"]
                )[0].cpu().numpy()
            fno_profiles.append(
                np.column_stack(
                    (np.exp(transformed[:, 0]), transformed[:, 1], np.exp(transformed[:, 2]))
                )
            )

        hybrid, hybrid_min, hybrid_max = profile_quantiles(hybrid_profiles)
        legacy, legacy_min, legacy_max = profile_quantiles(legacy_profiles)
        trusted, trusted_min, trusted_max = profile_quantiles(trusted_profiles)
        fno, fno_min, fno_max = profile_quantiles(fno_profiles)
        names = ("density", "velocity", "pressure")
        rows = []
        for index, coordinate in enumerate(x):
            row = {"x": coordinate}
            for component, name in enumerate(names):
                row.update(
                    {
                        f"exact_{name}": exact_profile[index, component],
                        f"piecewise_{name}": piecewise_profile[index, component],
                        f"fixed_mc_{name}": fixed_profile[index, component],
                        f"fine_hllc_{name}": fine_profile[index, component],
                        f"dkan_fv_{name}": hybrid[index, component],
                        f"dkan_fv_min_{name}": hybrid_min[index, component],
                        f"dkan_fv_max_{name}": hybrid_max[index, component],
                        f"legacy_dkan_fv_{name}": legacy[index, component],
                        f"legacy_dkan_fv_min_{name}": legacy_min[index, component],
                        f"legacy_dkan_fv_max_{name}": legacy_max[index, component],
                        f"dkan_tv2_{name}": trusted[index, component],
                        f"dkan_tv2_min_{name}": trusted_min[index, component],
                        f"dkan_tv2_max_{name}": trusted_max[index, component],
                        f"fno_{name}": fno[index, component],
                        f"fno_min_{name}": fno_min[index, component],
                        f"fno_max_{name}": fno_max[index, component],
                    }
                )
            rows.append(row)
        write_rows(output / f"profile_{case_id}.csv", rows)

    metadata = {
        "case_selection": {
            case_id: CASEBOOK_DESCRIPTIONS[case_id] for case_id in case_ids
        },
        "selection_rule": "Four in-distribution and four stress cases selected before plotting to cover representative, edge, extrapolation, collision, reverse-step, and failure-control regimes; selection was not optimized for the proposed method's error.",
        "six_models": [
            "FNO",
            "128-cell HLLC piecewise constant",
            "128-cell HLLC fixed MC",
            "narrow-distribution DKAN-FV",
            "engineering-distribution CSE-DKAN-FV",
            "512-cell HLLC",
        ],
        "profile_aggregation": "Pointwise median across three FNO seeds or three engineering DKAN-FV seeds; min-max seed envelope exported.",
    }
    (output / "profile_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_canonical(root, output)
    export_frozen_summaries(root, output)
    export_training_histories(root, output)
    export_casebook_metrics(root, output, CASEBOOK_IDS)
    export_profiles(
        root,
        output,
        CASEBOOK_IDS,
        [root / f"results/engineering_riemann_v1/fno/seed_{seed}.pt" for seed in (71001, 71002, 71003)],
        [root / f"results/final_float64_v1/euler_models/seed_{seed}/model.pt" for seed in (26001, 26002, 26003)],
        [root / f"results/engineering_riemann_v1/hybrid/seed_{seed}/model.pt" for seed in (72001, 72002, 72003)],
    )
    export_case_diagnostics(
        root,
        output,
        CASEBOOK_IDS,
        [root / f"results/engineering_riemann_v1/fno/seed_{seed}.pt" for seed in (71001, 71002, 71003)],
        [root / f"results/final_float64_v1/euler_models/seed_{seed}/model.pt" for seed in (26001, 26002, 26003)],
        [root / f"results/engineering_riemann_v1/hybrid/seed_{seed}/model.pt" for seed in (72001, 72002, 72003)],
    )
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
