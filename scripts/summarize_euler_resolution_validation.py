"""Summarize the frozen multiseed resolution-gated Euler validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_median(values: np.ndarray, count: int = 20000) -> list[float]:
    rng = np.random.default_rng(20260715)
    samples = np.empty(count, dtype=np.float64)
    for index in range(count):
        samples[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.protocol).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)

    expected_seeds = [int(value) for value in protocol["frozen_training_seeds"]]
    seed_directories = sorted(Path(args.root).glob("seed_*"))
    actual_seeds = [int(path.name.split("_")[-1]) for path in seed_directories]
    if actual_seeds != expected_seeds:
        raise ValueError(f"expected seeds {expected_seeds}, found {actual_seeds}")

    paired = []
    seed_aggregates = []
    for seed, directory in zip(actual_seeds, seed_directories, strict=True):
        seed_rows = []
        for path in sorted(directory.glob("v*.json")):
            with path.open("r", encoding="utf-8") as handle:
                benchmark = json.load(handle)
            for cells in protocol["benchmark"]["resolutions"]:
                fixed = next(
                    row
                    for row in benchmark["rows"]
                    if row["cells"] == cells
                    and row["method"] == "fixed_conservative_mc_subcell"
                )
                learned = next(
                    row
                    for row in benchmark["rows"]
                    if row["cells"] == cells
                    and row["method"]
                    == "resolution_gated_euler_jump_dkan_subcell"
                )
                gain = float(fixed["primitive_geometric_mean_l2"]) / float(
                    learned["primitive_geometric_mean_l2"]
                )
                rh_ratio = float(learned["shock_rh_residual_l2"]) / max(
                    float(fixed["shock_rh_residual_l2"]), 1.0e-15
                )
                row = {
                    "seed": seed,
                    "case_id": learned["case_id"],
                    "cells": cells,
                    "composite_l2_gain": gain,
                    "density_l2_gain": float(fixed["density_normalized_l2"])
                    / float(learned["density_normalized_l2"]),
                    "velocity_l2_gain": float(fixed["velocity_normalized_l2"])
                    / float(learned["velocity_normalized_l2"]),
                    "pressure_l2_gain": float(fixed["pressure_normalized_l2"])
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
                    "fixed_reconstruction_seconds": float(
                        fixed["reconstruction_seconds"]
                    ),
                    "learned_reconstruction_seconds": float(
                        learned["reconstruction_seconds"]
                    ),
                    "fv_solve_seconds": float(learned["fv_solve_seconds"]),
                    "pretraining_seconds": float(
                        benchmark["pretraining_generation_seconds"]
                    )
                    + float(benchmark["pretraining_seconds"]),
                }
                paired.append(row)
                seed_rows.append(row)
        gains = np.asarray([row["composite_l2_gain"] for row in seed_rows])
        seed_aggregates.append(
            {
                "seed": seed,
                "geometric_mean_composite_l2_gain": float(
                    np.exp(np.mean(np.log(gains)))
                ),
                "minimum_pair_gain": float(np.min(gains)),
                "maximum_pair_gain": float(np.max(gains)),
            }
        )

    aggregate_values = np.asarray(
        [row["geometric_mean_composite_l2_gain"] for row in seed_aggregates]
    )
    median_gain = float(np.median(aggregate_values))
    confidence_interval = bootstrap_median(aggregate_values)
    gates = {
        "primary_non_degrading": bool(
            median_gain >= 1.0 and confidence_interval[0] >= 0.95
        ),
        "all_pairs_within_5_percent_degradation": bool(
            all(row["composite_l2_gain"] >= 1.0 / 1.05 for row in paired)
        ),
        "hard_cell_conservation": bool(
            all(row["maximum_cell_average_change"] <= 3.0e-7 for row in paired)
        ),
        "positivity": bool(
            all(
                row["minimum_density"] > 0.0 and row["minimum_pressure"] > 0.0
                for row in paired
            )
        ),
        "shock_entropy": bool(
            all(row["shock_entropy_violation"] <= 1.0e-8 for row in paired)
        ),
        "rh_no_material_degradation": bool(
            all(row["rh_residual_ratio"] <= 1.20 for row in paired)
        ),
    }
    burgers_gain = float(
        protocol["unified_cross_equation_gate"]["burgers_frozen_gain"]
    )
    cross_equation_gain = float(np.sqrt(burgers_gain * median_gain))
    result = {
        "protocol": protocol["protocol"],
        "seed_count": len(actual_seeds),
        "paired_case_count": len(paired),
        "primary": {
            "median_seed_geometric_mean_gain": median_gain,
            "bootstrap_median_95ci": confidence_interval,
        },
        "gates": gates,
        "euler_branch_pass": bool(all(gates.values())),
        "cross_equation_gain": cross_equation_gain,
        "cross_equation_20_percent_gate": bool(cross_equation_gain >= 1.20),
        "worst_pair": min(paired, key=lambda row: row["composite_l2_gain"]),
        "maximum_rh_residual_ratio": float(
            max(row["rh_residual_ratio"] for row in paired)
        ),
        "maximum_cell_average_change": float(
            max(row["maximum_cell_average_change"] for row in paired)
        ),
        "minimum_density": float(min(row["minimum_density"] for row in paired)),
        "minimum_pressure": float(min(row["minimum_pressure"] for row in paired)),
        "maximum_shock_entropy_violation": float(
            max(row["shock_entropy_violation"] for row in paired)
        ),
        "median_pretraining_seconds": float(
            np.median([row["pretraining_seconds"] for row in paired])
        ),
        "median_learned_reconstruction_seconds": float(
            np.median([row["learned_reconstruction_seconds"] for row in paired])
        ),
        "median_fv_solve_seconds": float(
            np.median([row["fv_solve_seconds"] for row in paired])
        ),
        "seed_aggregates": seed_aggregates,
        "paired": paired,
        "overall_claim_allowed": False,
        "overall_claim_blockers": [
            "the final ten-seed float64 confirmation is incomplete",
            "the successful model is a supervised finite-volume hybrid, not a data-free PINN",
            "the 20% result is an equal-equation aggregate and not a per-equation Euler gain"
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
