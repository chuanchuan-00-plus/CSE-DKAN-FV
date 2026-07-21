"""Summarize post-confirmation hybrid ablations on the v3 cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometric(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(values))))


def summarize_seed_values(rows: list[dict]) -> dict:
    errors = np.asarray([row["geometric_error"] for row in rows])
    gains = np.asarray([row["geometric_gain_vs_fixed"] for row in rows])
    return {
        "median_geometric_error": float(np.median(errors)),
        "median_gain_vs_fixed": float(np.median(gains)),
        "minimum_pair_gain_vs_fixed": min(row["minimum_pair_gain"] for row in rows),
        "maximum_pair_error_ratio_vs_full": max(
            row["maximum_pair_error_ratio_vs_full"] for row in rows
        ),
        "seed_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = read(Path(args.protocol))
    v3 = Path(args.v3_root)
    ablation = Path(args.ablation_root)

    burgers_by_variant = {name: [] for name in protocol["burgers"]["variants"]}
    for seed in protocol["burgers"]["seeds"]:
        payloads = {
            "base": read(v3 / "burgers_validation" / f"seed_{seed}" / "benchmark.json"),
            "no_viscous_transition_gate": read(
                ablation / "burgers" / "no_viscous_gate" / f"seed_{seed}" / "benchmark.json"
            ),
            "no_tv_projection": read(
                ablation / "burgers" / "no_tv_projection" / f"seed_{seed}" / "benchmark.json"
            ),
        }
        groups = {}
        for label, payload in payloads.items():
            for row in payload["rows"]:
                key = (row["amplitude"], row["viscosity"], row["cells"])
                groups.setdefault(key, {}).setdefault(label, {})[row["method"]] = row
        per_variant = {name: [] for name in burgers_by_variant}
        fixed_errors = []
        full_errors = []
        for key in sorted(groups):
            base = groups[key]["base"]
            fixed = base["fixed_mc_subcell"]
            full = base["sensor_gated_dkan_subcell"]
            choices = {
                "full": full,
                "no_discontinuity_basis": fixed,
                "no_spatial_jump_condition": base["conservative_dkan_subcell"],
                "no_viscous_transition_gate": groups[key]["no_viscous_transition_gate"]["sensor_gated_dkan_subcell"],
                "no_tv_projection": groups[key]["no_tv_projection"]["sensor_gated_dkan_subcell"],
            }
            fixed_errors.append(fixed["normalized_l2"])
            full_errors.append(full["normalized_l2"])
            for name, row in choices.items():
                per_variant[name].append(row)
        for name, rows in per_variant.items():
            errors = [row["normalized_l2"] for row in rows]
            gains = [f / e for f, e in zip(fixed_errors, errors)]
            ratios = [e / f for e, f in zip(errors, full_errors)]
            burgers_by_variant[name].append(
                {
                    "seed": seed,
                    "geometric_error": geometric(errors),
                    "geometric_gain_vs_fixed": geometric(gains),
                    "minimum_pair_gain": min(gains),
                    "maximum_pair_error_ratio_vs_full": max(ratios),
                    "maximum_tv_excess": max(row["tv_excess"] for row in rows),
                    "maximum_overshoot_undershoot": max(
                        row["overshoot_undershoot"] for row in rows
                    ),
                    "maximum_cell_average_change": max(
                        row["maximum_cell_average_change"] for row in rows
                    ),
                }
            )

    euler_by_variant = {name: [] for name in protocol["euler"]["variants"]}
    for seed in protocol["euler"]["seeds"]:
        per_variant = {name: [] for name in euler_by_variant}
        fixed_errors = []
        full_errors = []
        for case in range(11, 21):
            case_id = f"f{case:02d}"
            payloads = {
                "base": read(v3 / "euler_validation" / f"seed_{seed}" / f"{case_id}.json"),
                "no_trust_region": read(
                    ablation / "euler" / "no_trust_region" / f"seed_{seed}" / f"{case_id}.json"
                ),
                "no_resolution_fallback": read(
                    ablation / "euler" / "no_resolution_fallback" / f"seed_{seed}" / f"{case_id}.json"
                ),
            }
            for cells in (100, 200, 400):
                methods = {}
                for label, payload in payloads.items():
                    methods[label] = {
                        row["method"]: row
                        for row in payload["rows"]
                        if row["cells"] == cells
                    }
                fixed = methods["base"]["fixed_conservative_mc_subcell"]
                full = methods["base"]["resolution_gated_euler_jump_dkan_subcell"]
                choices = {
                    "full": full,
                    "no_discontinuity_basis": fixed,
                    "no_trust_region": methods["no_trust_region"]["resolution_gated_euler_jump_dkan_subcell"],
                    "no_resolution_fallback": methods["no_resolution_fallback"]["resolution_gated_euler_jump_dkan_subcell"],
                }
                fixed_errors.append(fixed["primitive_geometric_mean_l2"])
                full_errors.append(full["primitive_geometric_mean_l2"])
                for name, row in choices.items():
                    per_variant[name].append((row, fixed))
        for name, pairs in per_variant.items():
            rows = [pair[0] for pair in pairs]
            errors = [row["primitive_geometric_mean_l2"] for row in rows]
            gains = [
                fixed["primitive_geometric_mean_l2"] / row["primitive_geometric_mean_l2"]
                for row, fixed in pairs
            ]
            ratios = [e / f for e, f in zip(errors, full_errors)]
            euler_by_variant[name].append(
                {
                    "seed": seed,
                    "geometric_error": geometric(errors),
                    "geometric_gain_vs_fixed": geometric(gains),
                    "minimum_pair_gain": min(gains),
                    "maximum_pair_error_ratio_vs_full": max(ratios),
                    "maximum_rh_ratio": max(
                        row["shock_rh_residual_l2"]
                        / max(fixed["shock_rh_residual_l2"], 1.0e-15)
                        for row, fixed in pairs
                    ),
                    "minimum_density": min(row["minimum_density"] for row in rows),
                    "minimum_pressure": min(row["minimum_pressure"] for row in rows),
                    "maximum_entropy_violation": max(
                        row["shock_entropy_violation"] for row in rows
                    ),
                    "maximum_cell_average_change": max(
                        row["maximum_cell_average_change"] for row in rows
                    ),
                }
            )

    result = {
        "protocol": protocol["protocol"],
        "status": "completed_descriptive",
        "burgers": {
            name: summarize_seed_values(rows) for name, rows in burgers_by_variant.items()
        },
        "euler": {
            name: summarize_seed_values(rows) for name, rows in euler_by_variant.items()
        },
        "claim_note": protocol["claim_boundary"],
        "entropy_note": protocol["entropy_note"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        equation: {
            name: {key: value for key, value in summary.items() if key != "seed_rows"}
            for name, summary in result[equation].items()
        }
        for equation in ("burgers", "euler")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
