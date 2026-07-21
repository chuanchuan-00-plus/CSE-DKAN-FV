"""Summarize a frozen multiseed sensor-gated DKAN validation protocol."""

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
    paths = sorted(Path(args.root).glob("seed_*/benchmark.json"))
    expected_seeds = [int(value) for value in protocol["frozen_models"]]
    actual_seeds = [int(path.parent.name.split("_")[-1]) for path in paths]
    if actual_seeds != expected_seeds:
        raise ValueError(f"expected seeds {expected_seeds}, found {actual_seeds}")

    paired = []
    seed_aggregates = []
    for path, seed in zip(paths, actual_seeds, strict=True):
        with path.open("r", encoding="utf-8") as handle:
            benchmark = json.load(handle)
        groups = {}
        for row in benchmark["rows"]:
            key = (float(row["amplitude"]), float(row["viscosity"]), int(row["cells"]))
            groups.setdefault(key, {})[row["method"]] = row
        seed_gains = []
        for (amplitude, viscosity, cells), methods in sorted(groups.items()):
            fixed = methods["fixed_mc_subcell"]
            learned = methods["sensor_gated_dkan_subcell"]
            gain = float(fixed["normalized_l2"]) / float(learned["normalized_l2"])
            seed_gains.append(gain)
            paired.append(
                {
                    "seed": seed,
                    "amplitude": amplitude,
                    "viscosity": viscosity,
                    "cells": cells,
                    "shock_formed": bool(learned["shock_formed"]),
                    "l2_gain_over_fixed_mc": gain,
                    "fixed_l2": float(fixed["normalized_l2"]),
                    "gated_dkan_l2": float(learned["normalized_l2"]),
                    "tv_excess": float(learned["tv_excess"]),
                    "overshoot_undershoot": float(learned["overshoot_undershoot"]),
                    "maximum_cell_average_change": float(
                        learned["maximum_cell_average_change"]
                    ),
                    "reconstruction_seconds": float(learned["reconstruction_seconds"]),
                    "fv_plus_reconstruction_seconds": float(learned["fv_solve_seconds"])
                    + float(learned["reconstruction_seconds"]),
                    "pretraining_seconds": float(benchmark["pretraining_seconds"])
                    + float(benchmark["pretraining_generation_seconds"]),
                }
            )
        seed_aggregates.append(
            {
                "seed": seed,
                "geometric_mean_l2_gain": float(np.exp(np.mean(np.log(seed_gains)))),
                "minimum_case_l2_gain": float(np.min(seed_gains)),
                "maximum_case_l2_gain": float(np.max(seed_gains)),
            }
        )

    aggregate_values = np.asarray(
        [row["geometric_mean_l2_gain"] for row in seed_aggregates], dtype=np.float64
    )
    worst = min(paired, key=lambda row: row["l2_gain_over_fixed_mc"])
    primary = {
        "median_seed_geometric_mean_l2_gain": float(np.median(aggregate_values)),
        "bootstrap_median_95ci": bootstrap_median(aggregate_values),
        "threshold": 1.20,
    }
    gates = {
        "primary_20_percent": bool(
            primary["median_seed_geometric_mean_l2_gain"] >= 1.20
            and primary["bootstrap_median_95ci"][0] > 1.00
        ),
        "all_seed_cases_within_5_percent_degradation": bool(
            all(row["l2_gain_over_fixed_mc"] >= 1.0 / 1.05 for row in paired)
        ),
        "hard_cell_conservation": bool(
            all(row["maximum_cell_average_change"] <= 2.0e-7 for row in paired)
        ),
        "no_new_extrema": bool(
            all(row["overshoot_undershoot"] <= 1.0e-8 for row in paired)
        ),
        "profile_tv_within_5_percent": bool(
            all(row["tv_excess"] <= 0.05 for row in paired)
        ),
    }
    result = {
        "protocol": protocol["protocol"],
        "seed_count": len(actual_seeds),
        "paired_case_count": len(paired),
        "primary": primary,
        "gates": gates,
        "burgers_branch_pass": bool(all(gates.values())),
        "worst_case": worst,
        "maximum_tv_excess": float(max(row["tv_excess"] for row in paired)),
        "maximum_cell_average_change": float(
            max(row["maximum_cell_average_change"] for row in paired)
        ),
        "maximum_overshoot_undershoot": float(
            max(row["overshoot_undershoot"] for row in paired)
        ),
        "median_pretraining_seconds": float(
            np.median([row["pretraining_seconds"] for row in paired])
        ),
        "median_fv_plus_reconstruction_seconds": float(
            np.median([row["fv_plus_reconstruction_seconds"] for row in paired])
        ),
        "seed_aggregates": seed_aggregates,
        "paired": paired,
        "overall_claim_allowed": False,
        "overall_claim_blockers": [
            "Euler conservative DKAN subcell validation is not complete",
            "the final ten-seed float64 protocol is not complete",
            "this is a WENO-supervised conservative hybrid rather than a data-free PINN"
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
