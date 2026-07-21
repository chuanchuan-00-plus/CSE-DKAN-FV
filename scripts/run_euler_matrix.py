"""Run Euler PINN variants sequentially for exclusive-GPU timing."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cse_dkan.euler_training import EulerTrainingConfig, train_euler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", nargs="+", default=["mlp_pinn", "dkan_pinn", "cw_dkan", "cse_dkan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-interior", type=int, default=256)
    parser.add_argument("--batch-initial", type=int, default=256)
    parser.add_argument("--batch-boundary", type=int, default=128)
    parser.add_argument("--batch-cv", type=int, default=32)
    parser.add_argument("--evaluation-points", type=int, default=1000)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--artificial-viscosity", type=float, default=0.005)
    arguments = parser.parse_args()
    base = EulerTrainingConfig(
        steps=arguments.steps,
        batch_interior=arguments.batch_interior,
        batch_initial=arguments.batch_initial,
        batch_boundary=arguments.batch_boundary,
        batch_control_volumes=arguments.batch_cv,
        evaluation_points=arguments.evaluation_points,
        dtype=arguments.dtype,
        artificial_viscosity=arguments.artificial_viscosity,
        log_every=max(1, arguments.steps // 10),
    )
    root = Path(arguments.output_root)
    summary = []
    for variant in arguments.variants:
        for seed in arguments.seeds:
            result = train_euler(replace(base, variant=variant, seed=seed), root / variant / f"seed_{seed}")
            summary.append({"variant": variant, "seed": seed, **result["metrics"]})
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
