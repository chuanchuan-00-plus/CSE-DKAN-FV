"""Five-seed Burgers development summary with explicit local-conservation gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_median(values: np.ndarray, count: int = 20000) -> list[float]:
    rng = np.random.default_rng(20260715)
    samples = np.empty(count)
    for index in range(count):
        samples[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-dkan", required=True)
    parser.add_argument("--seed0-cw", required=True)
    parser.add_argument("--seed0-cse", required=True)
    parser.add_argument("--multiseed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for variant, path in (
        ("dkan_pinn", args.seed0_dkan),
        ("cw_dkan", args.seed0_cw),
        ("cse_dkan", args.seed0_cse),
    ):
        row = load_rows(Path(path))[0]
        if row["variant"] != variant:
            raise ValueError(f"expected {variant} in {path}")
        rows.append(row)
    rows.extend(load_rows(Path(args.multiseed)))
    indexed = {(row["variant"], int(row["seed"])): row for row in rows}

    paired = []
    for seed in range(5):
        baselines = [indexed[(variant, seed)] for variant in ("dkan_pinn", "cw_dkan")]
        cse = indexed[("cse_dkan", seed)]

        def envelope(name: str) -> float:
            return min(float(row[name]) for row in baselines)

        gains = {
            "state_l2": envelope("normalized_l2") / float(cse["normalized_l2"]),
            "gradient_width": envelope("gradient_equivalent_width")
            / float(cse["gradient_equivalent_width"]),
            "local_cv": envelope("cv_balance_rmse") / float(cse["cv_balance_rmse"]),
            "shock_cv": envelope("shock_cv_balance_rmse")
            / float(cse["shock_cv_balance_rmse"]),
            "entropy": envelope("entropy_violation_mean")
            / float(cse["entropy_violation_mean"]),
            "mass_drift": envelope("max_mass_drift")
            / max(float(cse["max_mass_drift"]), 1.0e-12),
            "training_cost": envelope("training_seconds") / float(cse["training_seconds"]),
        }
        profile_gate = bool(
            float(cse["tv_excess"]) <= 0.02
            and float(cse["overshoot_undershoot"]) <= 0.02
            and float(cse["shock_width_failure"]) == 0.0
        )
        local_conservation_gate = bool(
            float(cse["cv_balance_rmse"]) <= 1.10 * envelope("cv_balance_rmse")
            and float(cse["shock_cv_balance_rmse"])
            <= 1.10 * envelope("shock_cv_balance_rmse")
        )
        paired.append(
            {
                "seed": seed,
                "gain": gains,
                "cse_tv_excess": float(cse["tv_excess"]),
                "cse_overshoot_undershoot": float(cse["overshoot_undershoot"]),
                "cse_width_10_90": float(cse["shock_width_10_90"]),
                "profile_gate": profile_gate,
                "local_conservation_gate": local_conservation_gate,
            }
        )

    result = {
        "status": "development_only",
        "baseline_definition": "per-seed per-metric oracle envelope of DKAN-PINN and CW-DKAN",
        "seed_count": 5,
        "paired": paired,
        "median_state_l2_gain": float(
            np.median([row["gain"]["state_l2"] for row in paired])
        ),
        "state_l2_gain_bootstrap_median_95ci": bootstrap_median(
            np.asarray([row["gain"]["state_l2"] for row in paired])
        ),
        "median_local_cv_gain": float(
            np.median([row["gain"]["local_cv"] for row in paired])
        ),
        "local_cv_gain_bootstrap_median_95ci": bootstrap_median(
            np.asarray([row["gain"]["local_cv"] for row in paired])
        ),
        "profile_gate_success_fraction": float(
            np.mean([row["profile_gate"] for row in paired])
        ),
        "local_conservation_gate_success_fraction": float(
            np.mean([row["local_conservation_gate"] for row in paired])
        ),
        "engineering_gate": False,
        "strong_20_percent_gate": False,
        "claim_allowed": False,
        "claim_blockers": [
            "independent local CV and shock-CV errors are worse than the neural envelope in every seed",
            "TV/overshoot profile gate fails in every CSE seed",
            "training cost is roughly three times the strongest neural baseline",
            "float64 ten-seed final protocol is incomplete",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
