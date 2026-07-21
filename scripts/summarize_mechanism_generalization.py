"""Create traceable CSV tables and aggregate statistics for the final suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from run_euler_mechanism_benchmark import load_protocol


LABELS = {
    "fno_operator": "FNO",
    "hllc_piecewise_constant": "128-cell HLLC-PC",
    "hllc_fixed_mc_subcell": "128-cell HLLC-MC",
    "physics_gated_dkan_fv_sod_neighbourhood": "Narrow DKAN-FV",
    "sensor_gated_dkan_fv_engineering_tv_trust_0.02": "CSE-DKAN-FV",
    "hllc_muscl_fine": "512-cell HLLC",
}


def median_rows(rows: list[dict], methods: list[str], case_ids: list[str]) -> list[dict]:
    output = []
    fields = (
        "density_normalized_l2",
        "velocity_normalized_l2",
        "pressure_normalized_l2",
        "primitive_rms_l2",
        "primitive_geometric_mean_l2",
        "maximum_relative_conservation_error",
        "minimum_density",
        "minimum_pressure",
        "density_tv_excess",
        "density_gradient_normalized_l1",
        "gradient_concentration_width_ratio",
        "online_seconds",
        "maximum_cell_average_change",
        "tv_trust_multiplier",
    )
    for case_id in case_ids:
        for method in methods:
            block = [row for row in rows if row["case_id"] == case_id and row["method"] == method]
            if not block:
                continue
            record = {
                "case_id": case_id,
                "method": method,
                "method_label": LABELS[method],
                "seed_count": len({row["model_seed"] for row in block if row["model_seed"] is not None}) or 1,
            }
            for field in fields:
                values = [row[field] for row in block if row.get(field) is not None]
                record[field] = statistics.median(values) if values else None
            output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--v1-results", required=True)
    parser.add_argument("--reference-convergence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    results_root = Path(args.results)
    raw = json.loads((results_root / "raw_results.json").read_text(encoding="utf-8"))
    v1 = json.loads(Path(args.v1_results).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    case_ids = [case["id"] for case in protocol["cases"]]
    methods = raw["methods"]
    final_rows = [row for row in raw["rows"] if abs(row["time_fraction"] - 1.0) < 1.0e-12]
    final_medians = median_rows(final_rows, methods, case_ids)
    write_csv(output / "mechanism_metrics.csv", final_medians)

    catalog = []
    metric_lookup = {(row["case_id"], row["method"]): row for row in final_medians}
    proposed = "sensor_gated_dkan_fv_engineering_tv_trust_0.02"
    fixed = "hllc_fixed_mc_subcell"
    for case in protocol["cases"]:
        case_id = case["id"]
        cse_error = metric_lookup[(case_id, proposed)]["primitive_rms_l2"]
        fixed_error = metric_lookup[(case_id, fixed)]["primitive_rms_l2"]
        fno_error = metric_lookup[(case_id, "fno_operator")]["primitive_rms_l2"]
        catalog.append(
            {
                "case_id": case_id,
                "case_group": case["group"],
                "kind": case["kind"],
                "boundary": case["boundary"],
                "final_time": case["time"],
                "mechanism": case["mechanism"],
                "purpose": case["purpose"],
                "fixed_over_cse_gain": fixed_error / cse_error,
                "fno_over_cse_gain": fno_error / cse_error,
            }
        )
    write_csv(output / "mechanism_case_catalog.csv", catalog)

    time_medians = []
    for case_id in case_ids:
        times = sorted({row["time"] for row in raw["rows"] if row["case_id"] == case_id})
        for physical_time in times:
            for method in methods:
                block = [
                    row for row in raw["rows"]
                    if row["case_id"] == case_id and row["method"] == method and abs(row["time"] - physical_time) < 1.0e-14
                ]
                time_medians.append(
                    {
                        "case_id": case_id,
                        "time": physical_time,
                        "time_fraction": block[0]["time_fraction"],
                        "method": method,
                        "method_label": LABELS[method],
                        "primitive_rms_l2": statistics.median(row["primitive_rms_l2"] for row in block),
                    }
                )
    write_csv(output / "mechanism_time_losses.csv", time_medians)

    maximum_tv_ratio = 0.0
    maximum_mean_change = 0.0
    minimum_density = math.inf
    minimum_pressure = math.inf
    activation = {}
    for case_id in case_ids:
        data = np.load(results_root / "trajectories" / f"{case_id}.npz")
        cse = data[proposed]
        mc = data[fixed]
        tv_ratio = []
        for time_index in range(cse.shape[0]):
            cse_tv = float(np.sum(np.abs(np.diff(cse[time_index, :, 0]))))
            mc_tv = float(np.sum(np.abs(np.diff(mc[time_index, :, 0]))))
            tv_ratio.append(cse_tv / max(mc_tv, 1.0e-15))
        maximum_tv_ratio = max(maximum_tv_ratio, max(tv_ratio))
        gate = data["cse_blend"]
        activation[case_id] = {
            "mean_final_gate": float(np.mean(gate[-1])),
            "fraction_final_gate_above_0.1": float(np.mean(gate[-1] > 0.1)),
            "maximum_tv_ratio_to_fixed_mc": max(tv_ratio),
        }
        profile_rows = []
        for point_index, coordinate in enumerate(data["x"]):
            record = {"x": float(coordinate)}
            for component_index, component in enumerate(("density", "velocity", "pressure")):
                record[f"reference_{component}"] = float(data["reference"][-1, point_index, component_index])
                for method in methods:
                    record[f"{method}_{component}"] = float(data[method][-1, point_index, component_index])
            profile_rows.append(record)
        write_csv(output / f"mechanism_profile_{case_id}.csv", profile_rows)

        trajectory_rows = []
        expanded_gate = np.repeat(data["cse_blend"], data["x"].size // data["cse_blend"].shape[1], axis=1)
        for time_index, physical_time in enumerate(data["times"]):
            for point_index, coordinate in enumerate(data["x"]):
                record = {
                    "time": float(physical_time),
                    "x": float(coordinate),
                    "reference_density": float(data["reference"][time_index, point_index, 0]),
                    "cse_blend": float(expanded_gate[time_index, point_index]),
                }
                for method in methods:
                    density = float(data[method][time_index, point_index, 0])
                    record[f"{method}_density"] = density
                    record[f"{method}_absolute_density_error"] = abs(
                        density - record["reference_density"]
                    )
                trajectory_rows.append(record)
        write_csv(output / f"mechanism_trajectory_{case_id}.csv", trajectory_rows)
    proposed_rows = [row for row in raw["rows"] if row["method"] == proposed]
    maximum_mean_change = max(row["maximum_cell_average_change"] for row in proposed_rows)
    minimum_density = min(row["minimum_density"] for row in proposed_rows)
    minimum_pressure = min(row["minimum_pressure"] for row in proposed_rows)

    gains = [row["fixed_over_cse_gain"] for row in catalog]
    fno_gains = [row["fno_over_cse_gain"] for row in catalog]
    v1_final = [row for row in v1["rows"] if abs(row["time_fraction"] - 1.0) < 1.0e-12]
    v1_proposed = "physics_gated_dkan_fv_engineering_tv_trust_0.02"
    sensor_ablation = []
    for case_id in case_ids:
        before = statistics.median(
            row["primitive_rms_l2"] for row in v1_final
            if row["case_id"] == case_id and row["method"] == v1_proposed
        )
        after = metric_lookup[(case_id, proposed)]["primitive_rms_l2"]
        sensor_ablation.append(
            {"case_id": case_id, "without_sensor": before, "with_sensor": after, "gain": before / after}
        )
    write_csv(output / "sensor_ablation.csv", sensor_ablation)

    convergence = json.loads(Path(args.reference_convergence).read_text(encoding="utf-8"))["rows"]
    write_csv(output / "mechanism_reference_convergence.csv", convergence)
    summary = {
        "case_count": len(case_ids),
        "same_backbone_geometric_mean_gain": geometric_mean(gains),
        "same_backbone_median_gain": statistics.median(gains),
        "same_backbone_wins": sum(value > 1.0 for value in gains),
        "fno_geometric_mean_gain": geometric_mean(fno_gains),
        "fno_minimum_case_gain": min(fno_gains),
        "maximum_parent_mean_change": maximum_mean_change,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "maximum_density_tv_ratio_to_fixed_mc": maximum_tv_ratio,
        "activation": activation,
        "reference_convergence": convergence,
    }
    (output / "mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
