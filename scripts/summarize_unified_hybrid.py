"""Combine frozen Burgers and Euler branch gates without changing their weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burgers", required=True)
    parser.add_argument("--euler", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    burgers = read_json(args.burgers)
    euler = read_json(args.euler)
    burgers_gain = float(
        burgers["primary"]["median_seed_geometric_mean_l2_gain"]
    )
    euler_gain = float(euler["primary"]["median_seed_geometric_mean_gain"])
    equal_equation_gain = float(np.sqrt(burgers_gain * euler_gain))
    engineering_gate = bool(
        burgers["burgers_branch_pass"]
        and euler["euler_branch_pass"]
        and equal_equation_gain >= 1.20
    )
    result = {
        "status": "five_seed_float32_hybrid_validation",
        "model_name": "wave-gated hard-conservative DKAN finite-volume hybrid",
        "burgers_branch_gain": burgers_gain,
        "euler_branch_gain": euler_gain,
        "equal_equation_geometric_mean_gain": equal_equation_gain,
        "equivalent_aggregate_error_reduction_fraction": 1.0 - 1.0 / equal_equation_gain,
        "engineering_20_percent_gate_pass": engineering_gate,
        "strong_final_claim_allowed": False,
        "claim_blockers": [
            "the final ten-seed float64 confirmation has not been run",
            "the successful architecture is supervised by WENO/exact Riemann data and coupled to finite volume",
            "direct CSE-DKAN-PINN did not pass its local conservation/entropy gates",
            "the equal-equation aggregate does not mean Euler alone improves by 20%",
            "2D benchmarks have not been implemented"
        ],
        "source_summaries": {
            "burgers": str(Path(args.burgers).resolve()),
            "euler": str(Path(args.euler).resolve())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
