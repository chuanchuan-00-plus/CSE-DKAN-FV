"""Aggregate held-out conservative subcell results across pretraining seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_median(values: np.ndarray, count: int = 20000) -> list[float]:
    rng = np.random.default_rng(20260715)
    samples = np.empty(count)
    for index in range(count):
        samples[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def read_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", required=True)
    parser.add_argument("--other-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [Path(args.seed0)] + sorted(Path(args.other_root).glob("seed_*/benchmark.json"))
    reference_width = {0.0: 0.0015278221237256667, 1.0e-3: 0.0044872536926890555}
    paired = []
    for path in paths:
        result = read_result(path)
        seed = 31415 if path == Path(args.seed0) else int(path.parent.name.split("_")[-1])
        for viscosity in (0.0, 1.0e-3):
            rows = [
                row
                for row in result["rows"]
                if float(row["viscosity"]) == viscosity and int(row["cells"]) == 128
            ]
            indexed = {row["method"]: row for row in rows}
            fixed = indexed["fixed_mc_subcell"]
            learned = indexed["conservative_dkan_subcell"]
            fixed_width_error = abs(
                float(fixed["shock_width_10_90"]) - reference_width[viscosity]
            )
            learned_width_error = abs(
                float(learned["shock_width_10_90"]) - reference_width[viscosity]
            )
            paired.append(
                {
                    "seed": seed,
                    "viscosity": viscosity,
                    "l2_gain_over_fixed_mc": float(fixed["normalized_l2"])
                    / float(learned["normalized_l2"]),
                    "l1_gain_over_fixed_mc": float(fixed["normalized_l1"])
                    / float(learned["normalized_l1"]),
                    "width_error_gain_over_fixed_mc": fixed_width_error
                    / max(learned_width_error, 1.0e-12),
                    "learned_l2": float(learned["normalized_l2"]),
                    "learned_tv_excess": float(learned["tv_excess"]),
                    "learned_overshoot_undershoot": float(
                        learned["overshoot_undershoot"]
                    ),
                    "maximum_cell_average_change": float(
                        learned["maximum_cell_average_change"]
                    ),
                    "profile_gate": bool(
                        float(learned["tv_excess"]) <= 0.02
                        and float(learned["overshoot_undershoot"]) <= 1.0e-8
                    ),
                    "hard_cell_conservation_gate": bool(
                        float(learned["maximum_cell_average_change"]) <= 2.0e-7
                    ),
                    "pretraining_seconds": float(result["pretraining_seconds"])
                    + float(result["pretraining_generation_seconds"]),
                    "fv_plus_reconstruction_seconds": float(learned["fv_solve_seconds"])
                    + float(learned["reconstruction_seconds"]),
                }
            )

    by_viscosity = {}
    for viscosity in (0.0, 1.0e-3):
        rows = [row for row in paired if row["viscosity"] == viscosity]
        gains = np.asarray([row["l2_gain_over_fixed_mc"] for row in rows])
        by_viscosity[str(viscosity)] = {
            "seed_count": len(rows),
            "median_l2_gain_over_fixed_mc": float(np.median(gains)),
            "l2_gain_bootstrap_median_95ci": bootstrap_median(gains),
            "median_learned_l2": float(np.median([row["learned_l2"] for row in rows])),
            "profile_gate_success_fraction": float(
                np.mean([row["profile_gate"] for row in rows])
            ),
            "hard_cell_conservation_success_fraction": float(
                np.mean([row["hard_cell_conservation_gate"] for row in rows])
            ),
        }
    result = {
        "status": "development_multiseed_held_out_target",
        "method": "TVD finite-volume cell averages plus zero-mean bounded DKAN subcells",
        "paired": paired,
        "summary": by_viscosity,
        "claim_allowed": False,
        "claim_blockers": [
            "Euler conservative DKAN subcell reconstruction is not implemented",
            "ten-seed float64 final protocol is incomplete",
            "the subcell DKAN is WENO-supervised and is a hybrid conservative model, not a data-free PINN",
            "amortized pretraining cost and unseen parameter families require broader validation",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
