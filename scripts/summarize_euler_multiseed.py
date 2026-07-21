"""Conservative paired Gate summary for the five-seed Euler development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def geometric_mean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-baselines", required=True)
    parser.add_argument("--seed0-cse", required=True)
    parser.add_argument("--multiseed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    rows = load_rows(Path(args.seed0_baselines))
    with Path(args.seed0_cse).open("r", encoding="utf-8") as handle:
        cse0 = json.load(handle)
    rows.append({"variant": "cse_dkan", "seed": 0, **cse0["metrics"]})
    rows.extend(load_rows(Path(args.multiseed)))
    indexed = {(row["variant"], int(row["seed"])): row for row in rows}

    weights = {
        "state": 0.30,
        "resolution": 0.20,
        "conservation": 0.15,
        "entropy": 0.10,
        "robustness": 0.10,
        "training": 0.10,
        "inference": 0.05,
    }
    eps = {
        "state": 1.0e-6,
        "resolution": 1.0e-4,
        "conservation": 1.0e-6,
        "entropy": 1.0e-6,
        "training": 1.0e-3,
        "inference": 1.0e-6,
    }
    paired = []
    for seed in range(5):
        baselines = [indexed[(variant, seed)] for variant in ("dkan_pinn", "cw_dkan")]
        cse = indexed[("cse_dkan", seed)]

        def envelope(name: str) -> float:
            return min(float(row[name]) for row in baselines)

        baseline_groups = {
            "state": geometric_mean(
                [envelope("density_l2"), envelope("velocity_l2"), envelope("pressure_l2")]
            ),
            "resolution": geometric_mean(
                [
                    envelope("shock_location_error") + eps["resolution"],
                    envelope("contact_location_error") + eps["resolution"],
                    envelope("shock_width_10_90") + eps["resolution"],
                    envelope("contact_width_10_90") + eps["resolution"],
                ]
            ),
            "conservation": geometric_mean(
                [
                    envelope("max_mass_balance_error") + eps["conservation"],
                    envelope("max_momentum_balance_error") + eps["conservation"],
                    envelope("max_energy_balance_error") + eps["conservation"],
                ]
            ),
            "entropy": envelope("shock_entropy_violation") + eps["entropy"],
            "robustness": 1.0,
            "training": envelope("training_seconds") + eps["training"],
            "inference": envelope("inference_seconds") + eps["inference"],
        }
        cse_groups = {
            "state": geometric_mean(
                [float(cse["density_l2"]), float(cse["velocity_l2"]), float(cse["pressure_l2"])]
            ),
            "resolution": geometric_mean(
                [
                    float(cse["shock_location_error"]) + eps["resolution"],
                    float(cse["contact_location_error"]) + eps["resolution"],
                    float(cse["shock_width_10_90"]) + eps["resolution"],
                    float(cse["contact_width_10_90"]) + eps["resolution"],
                ]
            ),
            "conservation": geometric_mean(
                [
                    float(cse["max_mass_balance_error"]) + eps["conservation"],
                    float(cse["max_momentum_balance_error"]) + eps["conservation"],
                    float(cse["max_energy_balance_error"]) + eps["conservation"],
                ]
            ),
            "entropy": float(cse["shock_entropy_violation"]) + eps["entropy"],
            "robustness": 1.0,
            "training": float(cse["training_seconds"]) + eps["training"],
            "inference": float(cse["inference_seconds"]) + eps["inference"],
        }
        ratios = {
            name: baseline_groups[name] / cse_groups[name]
            for name in weights
        }
        composite = float(
            np.exp(sum(weights[name] * np.log(ratios[name]) for name in weights))
        )
        primary_degradation = max(
            cse_groups["state"] / baseline_groups["state"] - 1.0,
            cse_groups["conservation"] / baseline_groups["conservation"] - 1.0,
        )
        paired.append(
            {
                "seed": seed,
                "baseline_envelope": baseline_groups,
                "cse": cse_groups,
                "gain_by_group": ratios,
                "composite_gain": composite,
                "max_state_or_conservation_degradation": primary_degradation,
            }
        )

    gains = np.asarray([row["composite_gain"] for row in paired])
    rng = np.random.default_rng(20260715)
    bootstrap = np.empty(args.bootstrap)
    for index in range(args.bootstrap):
        bootstrap[index] = np.median(rng.choice(gains, size=gains.size, replace=True))
    ci = np.quantile(bootstrap, [0.025, 0.975])
    result = {
        "status": "development_only_exact_sod_geometry_prior",
        "baseline_definition": "per-seed per-metric oracle envelope of DKAN-PINN and CW-DKAN",
        "seed_count": 5,
        "paired": paired,
        "median_composite_gain": float(np.median(gains)),
        "bootstrap_median_95ci": [float(ci[0]), float(ci[1])],
        "state_improvement_success_fraction": float(
            np.mean([row["gain_by_group"]["state"] > 1.0 for row in paired])
        ),
        "no_primary_degradation_success_fraction": float(
            np.mean([row["max_state_or_conservation_degradation"] <= 0.10 for row in paired])
        ),
        "engineering_gate_euler_only": bool(
            np.median(gains) >= 1.20
            and ci[0] > 1.0
            and all(row["max_state_or_conservation_degradation"] <= 0.10 for row in paired)
        ),
        "strong_20_percent_gate_euler_only": bool(
            ci[0] >= 1.20
            and all(row["max_state_or_conservation_degradation"] <= 0.10 for row in paired)
        ),
        "claim_allowed": False,
        "claim_blockers": [
            "two of five CSE seeds show high state error",
            "exact Sod shock/contact geometry and star-state traces are frozen priors",
            "all required 1D tasks and ten final seeds are not complete",
            "float64 final-budget comparison is not complete",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
