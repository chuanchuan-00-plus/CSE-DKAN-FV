"""Mechanical audit and summary for final_1d_float64_v1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_median(values: np.ndarray, count: int = 50000) -> list[float]:
    rng = np.random.default_rng(20260716)
    samples = np.empty(count, dtype=np.float64)
    for index in range(count):
        samples[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def audit_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    floating = [
        value for value in checkpoint["model_state"].values() if value.is_floating_point()
    ]
    return {
        "path": str(path.resolve()),
        "metadata_dtype": checkpoint.get("dtype"),
        "all_floating_tensors_float64": bool(
            floating and all(value.dtype == torch.float64 for value in floating)
        ),
        "floating_tensor_count": len(floating),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--burgers-root", required=True)
    parser.add_argument("--euler-root", required=True)
    parser.add_argument("--reference-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = read_json(Path(args.protocol))
    reference_audit = read_json(Path(args.reference_audit))
    burgers_seed_rows = []
    burgers_pairs = []
    dtype_audits = []
    expected_burgers_seeds = protocol["burgers"]["training_seeds"]
    expected_amplitudes = protocol["burgers"]["blind_benchmark"]["amplitudes"]
    expected_viscosities = protocol["burgers"]["blind_benchmark"]["viscosities"]
    expected_burgers_keys = {
        (float(amplitude), float(viscosity), 128)
        for amplitude in expected_amplitudes
        for viscosity in expected_viscosities
    }
    for seed in expected_burgers_seeds:
        path = Path(args.burgers_root) / f"seed_{seed}" / "benchmark.json"
        benchmark = read_json(path)
        if benchmark.get("dtype") != "float64":
            raise ValueError(f"non-float64 Burgers benchmark: {path}")
        checkpoint_path = Path(benchmark["checkpoint"])
        dtype_audits.append(audit_checkpoint(checkpoint_path))
        groups = {}
        for row in benchmark["rows"]:
            key = (float(row["amplitude"]), float(row["viscosity"]), int(row["cells"]))
            groups.setdefault(key, {})[row["method"]] = row
        if set(groups) != expected_burgers_keys:
            raise ValueError(f"Burgers parameter coverage mismatch for seed {seed}")
        gains = []
        for key in sorted(groups):
            fixed = groups[key]["fixed_mc_subcell"]
            learned = groups[key]["sensor_gated_dkan_subcell"]
            gain = float(fixed["normalized_l2"]) / float(learned["normalized_l2"])
            gains.append(gain)
            burgers_pairs.append(
                {
                    "seed": seed,
                    "amplitude": key[0],
                    "viscosity": key[1],
                    "cells": key[2],
                    "gain": gain,
                    "fixed_l2": float(fixed["normalized_l2"]),
                    "learned_l2": float(learned["normalized_l2"]),
                    "tv_excess": float(learned["tv_excess"]),
                    "overshoot_undershoot": float(learned["overshoot_undershoot"]),
                    "maximum_cell_average_change": float(
                        learned["maximum_cell_average_change"]
                    ),
                    "reference_cells": int(learned["reference_cells"]),
                    "fv_seconds": float(learned["fv_solve_seconds"]),
                    "reconstruction_seconds": float(learned["reconstruction_seconds"]),
                    "pretraining_seconds": float(benchmark["pretraining_generation_seconds"])
                    + float(benchmark["pretraining_seconds"]),
                }
            )
        burgers_seed_rows.append(
            {
                "seed": seed,
                "geometric_mean_gain": float(np.exp(np.mean(np.log(gains)))),
                "minimum_pair_gain": float(np.min(gains)),
                "maximum_pair_gain": float(np.max(gains)),
            }
        )

    euler_seed_rows = []
    euler_pairs = []
    expected_euler_seeds = protocol["euler"]["training_seeds"]
    expected_euler_cases = {
        item["id"]: item for item in protocol["euler"]["blind_benchmark"]["cases"]
    }
    expected_resolutions = protocol["euler"]["blind_benchmark"]["resolutions"]
    for seed in expected_euler_seeds:
        checkpoint_audited = False
        gains = []
        for case_id, expected_case in expected_euler_cases.items():
            path = Path(args.euler_root) / f"seed_{seed}" / f"{case_id}.json"
            benchmark = read_json(path)
            if benchmark.get("dtype") != "float64":
                raise ValueError(f"non-float64 Euler benchmark: {path}")
            if not checkpoint_audited:
                dtype_audits.append(audit_checkpoint(Path(benchmark["checkpoint"])))
                checkpoint_audited = True
            for cells in expected_resolutions:
                fixed = next(
                    row
                    for row in benchmark["rows"]
                    if row["method"] == "fixed_conservative_mc_subcell"
                    and int(row["cells"]) == cells
                )
                learned = next(
                    row
                    for row in benchmark["rows"]
                    if row["method"] == "resolution_gated_euler_jump_dkan_subcell"
                    and int(row["cells"]) == cells
                )
                if learned["left_state"] != expected_case["left"] or learned[
                    "right_state"
                ] != expected_case["right"]:
                    raise ValueError(f"Euler state mismatch for {case_id}")
                gain = float(fixed["primitive_geometric_mean_l2"]) / float(
                    learned["primitive_geometric_mean_l2"]
                )
                rh_ratio = float(learned["shock_rh_residual_l2"]) / max(
                    float(fixed["shock_rh_residual_l2"]), 1.0e-15
                )
                gains.append(gain)
                euler_pairs.append(
                    {
                        "seed": seed,
                        "case_id": case_id,
                        "cells": cells,
                        "gain": gain,
                        "density_gain": float(fixed["density_normalized_l2"])
                        / float(learned["density_normalized_l2"]),
                        "velocity_gain": float(fixed["velocity_normalized_l2"])
                        / float(learned["velocity_normalized_l2"]),
                        "pressure_gain": float(fixed["pressure_normalized_l2"])
                        / float(learned["pressure_normalized_l2"]),
                        "rh_residual_ratio": rh_ratio,
                        "maximum_cell_average_change": float(
                            learned["maximum_cell_average_change"]
                        ),
                        "minimum_density": float(learned["minimum_density"]),
                        "minimum_pressure": float(learned["minimum_pressure"]),
                        "shock_entropy_violation": float(
                            learned["shock_entropy_violation"]
                        ),
                        "fv_seconds": float(learned["fv_solve_seconds"]),
                        "reconstruction_seconds": float(
                            learned["reconstruction_seconds"]
                        ),
                        "pretraining_seconds": float(
                            benchmark["pretraining_generation_seconds"]
                        )
                        + float(benchmark["pretraining_oracle_seconds"])
                        + float(benchmark["pretraining_seconds"]),
                    }
                )
        euler_seed_rows.append(
            {
                "seed": seed,
                "geometric_mean_gain": float(np.exp(np.mean(np.log(gains)))),
                "minimum_pair_gain": float(np.min(gains)),
                "maximum_pair_gain": float(np.max(gains)),
            }
        )

    if not all(
        audit["metadata_dtype"] == "float64"
        and audit["all_floating_tensors_float64"]
        for audit in dtype_audits
    ):
        raise ValueError("checkpoint dtype audit failed")
    burgers_values = np.asarray(
        [row["geometric_mean_gain"] for row in burgers_seed_rows]
    )
    euler_values = np.asarray([row["geometric_mean_gain"] for row in euler_seed_rows])
    cross_values = np.sqrt(burgers_values * euler_values)
    burgers_ci = bootstrap_median(burgers_values)
    euler_ci = bootstrap_median(euler_values)
    cross_ci = bootstrap_median(cross_values)
    burgers_gates = {
        "median_gain": bool(float(np.median(burgers_values)) >= 1.20),
        "bootstrap_lower": bool(burgers_ci[0] > 1.00),
        "every_pair_no_material_degradation": bool(
            all(row["gain"] >= 1.0 / 1.05 for row in burgers_pairs)
        ),
        "hard_cell_conservation": bool(
            all(row["maximum_cell_average_change"] <= 1.0e-12 for row in burgers_pairs)
        ),
        "no_new_extrema": bool(
            all(row["overshoot_undershoot"] <= 1.0e-10 for row in burgers_pairs)
        ),
        "profile_tv": bool(all(row["tv_excess"] <= 0.05 for row in burgers_pairs)),
        "reference_convergence": bool(reference_audit["all_pass"]),
    }
    euler_gates = {
        "median_gain": bool(float(np.median(euler_values)) >= 1.00),
        "bootstrap_lower": bool(euler_ci[0] >= 0.95),
        "every_pair_no_material_degradation": bool(
            all(row["gain"] >= 1.0 / 1.05 for row in euler_pairs)
        ),
        "hard_cell_conservation": bool(
            all(row["maximum_cell_average_change"] <= 1.0e-12 for row in euler_pairs)
        ),
        "positivity": bool(
            all(
                row["minimum_density"] > 0.0 and row["minimum_pressure"] > 0.0
                for row in euler_pairs
            )
        ),
        "shock_entropy": bool(
            all(row["shock_entropy_violation"] <= 1.0e-10 for row in euler_pairs)
        ),
        "rh_no_material_degradation": bool(
            all(row["rh_residual_ratio"] <= 1.20 for row in euler_pairs)
        ),
        "fine_grid_exact_fallback": bool(
            all(
                abs(row["gain"] - 1.0) <= 1.0e-12
                for row in euler_pairs
                if row["cells"] == 400
            )
        ),
    }
    cross_gates = {
        "engineering_median": bool(float(np.median(cross_values)) >= 1.20),
        "engineering_bootstrap_lower": bool(cross_ci[0] > 1.00),
        "strong_20_percent_bootstrap_lower": bool(cross_ci[0] >= 1.20),
    }
    protocol_pass = bool(
        all(burgers_gates.values())
        and all(euler_gates.values())
        and cross_gates["engineering_median"]
        and cross_gates["engineering_bootstrap_lower"]
    )
    result = {
        "protocol": protocol["protocol"],
        "status": "passed" if protocol_pass else "failed",
        "dtype_checkpoint_count": len(dtype_audits),
        "dtype_audit_pass": True,
        "burgers": {
            "seed_count": len(burgers_seed_rows),
            "pair_count": len(burgers_pairs),
            "median_gain": float(np.median(burgers_values)),
            "bootstrap_median_95ci": burgers_ci,
            "gates": burgers_gates,
            "worst_pair": min(burgers_pairs, key=lambda row: row["gain"]),
            "maximum_tv_excess": max(row["tv_excess"] for row in burgers_pairs),
            "maximum_cell_average_change": max(
                row["maximum_cell_average_change"] for row in burgers_pairs
            ),
            "maximum_overshoot_undershoot": max(
                row["overshoot_undershoot"] for row in burgers_pairs
            ),
            "median_pretraining_seconds": float(
                np.median([row["pretraining_seconds"] for row in burgers_pairs])
            ),
            "seed_aggregates": burgers_seed_rows,
        },
        "euler": {
            "seed_count": len(euler_seed_rows),
            "pair_count": len(euler_pairs),
            "median_gain": float(np.median(euler_values)),
            "bootstrap_median_95ci": euler_ci,
            "gates": euler_gates,
            "worst_pair": min(euler_pairs, key=lambda row: row["gain"]),
            "maximum_rh_residual_ratio": max(
                row["rh_residual_ratio"] for row in euler_pairs
            ),
            "maximum_cell_average_change": max(
                row["maximum_cell_average_change"] for row in euler_pairs
            ),
            "minimum_density": min(row["minimum_density"] for row in euler_pairs),
            "minimum_pressure": min(row["minimum_pressure"] for row in euler_pairs),
            "maximum_shock_entropy_violation": max(
                row["shock_entropy_violation"] for row in euler_pairs
            ),
            "median_pretraining_seconds": float(
                np.median([row["pretraining_seconds"] for row in euler_pairs])
            ),
            "seed_aggregates": euler_seed_rows,
        },
        "cross_equation": {
            "per_seed_gains": [float(value) for value in cross_values],
            "median_gain": float(np.median(cross_values)),
            "bootstrap_median_95ci": cross_ci,
            "gates": cross_gates,
        },
        "final_protocol_pass": protocol_pass,
        "strong_20_percent_claim_allowed": bool(
            protocol_pass and cross_gates["strong_20_percent_bootstrap_lower"]
        ),
        "direct_cse_dkan_pinn_claim_allowed": False,
        "claim_note": "The direct PINN branch remains falsified; this protocol evaluates a supervised finite-volume hybrid.",
        "dtype_audits": dtype_audits,
        "burgers_pairs": burgers_pairs,
        "euler_pairs": euler_pairs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in {"burgers_pairs", "euler_pairs", "dtype_audits"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
