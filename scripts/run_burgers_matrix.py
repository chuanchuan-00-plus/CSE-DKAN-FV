"""Run Burgers variants sequentially so timing comparisons use an exclusive GPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cse_dkan.training import BurgersTrainingConfig, train_burgers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--variants", nargs="+", default=["mlp_pinn", "dkan_pinn", "cw_dkan", "cse_dkan"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-interior", type=int, default=512)
    parser.add_argument("--batch-cv", type=int, default=64)
    parser.add_argument("--batch-rh", type=int, default=64)
    parser.add_argument("--reference-cells", type=int, default=2048)
    parser.add_argument("--evaluation-points", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--viscosity", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--mean", type=float, default=0.2)
    parser.add_argument("--final-time", type=float, default=0.6)
    parser.add_argument("--artificial-viscosity", type=float, default=0.01)
    arguments = parser.parse_args()

    output_root = Path(arguments.output_root)
    base = BurgersTrainingConfig(
        steps=arguments.steps,
        batch_interior=arguments.batch_interior,
        batch_control_volumes=arguments.batch_cv,
        batch_rh=arguments.batch_rh,
        reference_cells=arguments.reference_cells,
        evaluation_points=arguments.evaluation_points,
        dtype=arguments.dtype,
        viscosity=arguments.viscosity,
        amplitude=arguments.amplitude,
        mean=arguments.mean,
        final_time=arguments.final_time,
        artificial_viscosity=arguments.artificial_viscosity,
        log_every=max(1, arguments.steps // 10),
    )
    summary = []
    for variant in arguments.variants:
        for seed in arguments.seeds:
            config = replace(base, variant=variant, seed=seed)
            output = output_root / variant / f"seed_{seed}"
            result = train_burgers(config, output)
            summary.append({"variant": variant, "seed": seed, **result["metrics"]})

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
