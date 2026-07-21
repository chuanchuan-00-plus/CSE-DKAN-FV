"""Audit and summarize canonical PINN baseline runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def statistics(rows: list[dict], names: list[str]) -> dict:
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hybrid-root")
    args = parser.parse_args()

    protocol = read(Path(args.protocol))
    root = Path(args.root)
    seeds = protocol["common"]["seeds"]
    variants = list(protocol["common"]["variants"])
    raw = {"burgers": [], "euler": []}
    expected_steps = {
        equation: protocol[equation]["training"]["steps"]
        for equation in ("burgers", "euler")
    }
    parameter_counts = {"burgers": set(), "euler": set()}
    for equation in ("burgers", "euler"):
        for seed in seeds:
            for variant in variants:
                path = (
                    root
                    / equation
                    / f"batch_seed_{seed}"
                    / variant
                    / f"seed_{seed}"
                    / "result.json"
                )
                payload = read(path)
                config = payload["config"]
                if config["variant"] != variant or config["seed"] != seed:
                    raise ValueError(f"variant/seed mismatch: {path}")
                if config["dtype"] != "float64" or config["steps"] != expected_steps[equation]:
                    raise ValueError(f"numeric budget mismatch: {path}")
                metrics = payload["metrics"]
                parameters = int(metrics["optimized_parameters"])
                parameter_counts[equation].add(parameters)
                row = {"variant": variant, "seed": seed, **metrics}
                if equation == "euler":
                    row["primitive_geometric_mean_l2"] = math.exp(
                        np.mean(
                            np.log(
                                [
                                    metrics["density_l2"],
                                    metrics["velocity_l2"],
                                    metrics["pressure_l2"],
                                ]
                            )
                        )
                    )
                raw[equation].append(row)
    if any(len(counts) != 1 for counts in parameter_counts.values()):
        raise ValueError("PINN parameter counts are not exactly matched")

    burgers_metrics = [
        "normalized_l2",
        "shock_width_10_90",
        "shock_location_error",
        "mass_error",
        "max_mass_drift",
        "cv_balance_rmse",
        "entropy_violation_max",
        "training_seconds",
        "inference_seconds",
    ]
    euler_metrics = [
        "primitive_geometric_mean_l2",
        "density_l2",
        "velocity_l2",
        "pressure_l2",
        "shock_width_10_90",
        "shock_location_error",
        "max_mass_balance_error",
        "max_momentum_balance_error",
        "max_energy_balance_error",
        "shock_entropy_violation",
        "minimum_density",
        "minimum_pressure",
        "training_seconds",
        "inference_seconds",
    ]
    result = {
        "protocol": protocol["protocol"],
        "status": "completed_descriptive",
        "seed_count": len(seeds),
        "configuration_audit_pass": True,
        "optimized_parameters": {
            equation: next(iter(counts)) for equation, counts in parameter_counts.items()
        },
        "burgers": {
            variant: statistics(
                [row for row in raw["burgers"] if row["variant"] == variant],
                burgers_metrics,
            )
            for variant in variants
        },
        "euler": {
            variant: statistics(
                [row for row in raw["euler"] if row["variant"] == variant],
                euler_metrics,
            )
            for variant in variants
        },
        "raw": raw,
        "claim_note": "Canonical single-problem descriptive baselines; separate from the v3 parameter-scan confirmation.",
    }
    if args.hybrid_root:
        hybrid_root = Path(args.hybrid_root)
        burgers_hybrid = []
        for path in sorted((hybrid_root / "burgers").glob("seed_*/benchmark.json")):
            payload = read(path)
            methods = {row["method"]: row for row in payload["rows"]}
            fixed = methods["fixed_mc_subcell"]
            learned = methods["sensor_gated_dkan_subcell"]
            burgers_hybrid.append(
                {
                    "fixed_l2": fixed["normalized_l2"],
                    "hybrid_l2": learned["normalized_l2"],
                    "gain": fixed["normalized_l2"] / learned["normalized_l2"],
                    "shock_width_10_90": learned["shock_width_10_90"],
                    "fv_seconds": learned["fv_solve_seconds"],
                    "reconstruction_seconds": learned["reconstruction_seconds"],
                    "pretraining_seconds": payload["pretraining_generation_seconds"]
                    + payload["pretraining_seconds"],
                }
            )
        euler_hybrid = []
        for path in sorted((hybrid_root / "euler").glob("seed_*/sod.json")):
            payload = read(path)
            methods = {
                row["method"]: row for row in payload["rows"] if row["cells"] == 200
            }
            fixed = methods["fixed_conservative_mc_subcell"]
            learned = methods["resolution_gated_euler_jump_dkan_subcell"]
            euler_hybrid.append(
                {
                    "fixed_l2": fixed["primitive_geometric_mean_l2"],
                    "hybrid_l2": learned["primitive_geometric_mean_l2"],
                    "gain": fixed["primitive_geometric_mean_l2"]
                    / learned["primitive_geometric_mean_l2"],
                    "shock_width_10_90": learned["density_shock_width_10_90"],
                    "fv_seconds": learned["fv_solve_seconds"],
                    "reconstruction_seconds": learned["reconstruction_seconds"],
                    "pretraining_seconds": payload["pretraining_generation_seconds"]
                    + payload["pretraining_oracle_seconds"]
                    + payload["pretraining_seconds"],
                }
            )
        result["hybrid_context"] = {
            "burgers": statistics(
                burgers_hybrid,
                [
                    "fixed_l2", "hybrid_l2", "gain", "shock_width_10_90",
                    "fv_seconds", "reconstruction_seconds", "pretraining_seconds",
                ],
            ),
            "euler": statistics(
                euler_hybrid,
                [
                    "fixed_l2", "hybrid_l2", "gain", "shock_width_10_90",
                    "fv_seconds", "reconstruction_seconds", "pretraining_seconds",
                ],
            ),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    compact = {key: value for key, value in result.items() if key != "raw"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
