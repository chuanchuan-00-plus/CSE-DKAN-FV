"""Prospective Euler CSE ablations under an identical training budget."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cse_dkan.euler_training import EulerTrainingConfig, train_euler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["full", "no_trace", "no_rh", "no_entropy", "learned_geometry"],
    )
    arguments = parser.parse_args()
    base = EulerTrainingConfig(
        variant="cse_dkan",
        seed=arguments.seed,
        steps=arguments.steps,
        batch_interior=256,
        batch_control_volumes=32,
        evaluation_points=1000,
        log_every=max(1, arguments.steps // 5),
    )
    configurations = {
        "full": base,
        "no_trace": replace(base, weight_trace=0.0),
        "no_rh": replace(base, weight_rh=0.0, weight_contact=0.0),
        "no_entropy": replace(base, weight_entropy=0.0, weight_shock_entropy=0.0),
        "learned_geometry": replace(base, freeze_wave_geometry=False),
    }
    root = Path(arguments.output_root)
    summary = []
    for name in arguments.ablations:
        if name not in configurations:
            raise ValueError(f"unknown ablation: {name}")
        result = train_euler(configurations[name], root / name / f"seed_{arguments.seed}")
        summary.append({"ablation": name, "seed": arguments.seed, **result["metrics"]})
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
