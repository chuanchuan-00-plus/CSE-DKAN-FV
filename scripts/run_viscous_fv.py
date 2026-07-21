"""Benchmark the conservative viscous Burgers finite-volume backbone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.finite_volume import ConservativeBurgersFV
from cse_dkan.metrics import (
    gradient_equivalent_width,
    normalized_lp,
    overshoot_undershoot,
    shock_location_from_gradient,
    shock_width_10_90,
    total_variation_excess,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--cells", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--reference-cells", type=int, default=4096)
    args = parser.parse_args()

    problem = PeriodicBurgersProblem(viscosity=args.viscosity)
    reference_x, snapshots = problem.reference(args.reference_cells)
    reference_u = snapshots[problem.final_time]
    expected_shock = problem.origin + problem.mean * problem.final_time
    window = (expected_shock - 0.15, expected_shock + 0.15)
    rows: list[dict[str, float | int | str]] = []
    for limiter in ("first_order", "tvd"):
        for cells in args.cells:
            x = problem.x_left + (np.arange(cells) + 0.5) * (
                problem.x_right - problem.x_left
            ) / cells
            initial_np = problem.initial_numpy(x)
            initial = torch.tensor(initial_np, dtype=torch.float64)
            solver = ConservativeBurgersFV(
                n_cells=cells, limiter=limiter, viscosity=problem.viscosity
            )
            started = time.perf_counter()
            prediction = solver.solve(initial, problem.final_time).detach().numpy()
            runtime = time.perf_counter() - started
            reference = np.interp(
                x,
                reference_x,
                reference_u,
                period=problem.x_right - problem.x_left,
            )
            reference_location = shock_location_from_gradient(x, reference, window)
            location = shock_location_from_gradient(x, prediction, window)
            left_state = float(np.interp(expected_shock - 0.02, reference_x, reference_u))
            right_state = float(np.interp(expected_shock + 0.02, reference_x, reference_u))
            rows.append(
                {
                    "method": f"viscous_godunov_{limiter}",
                    "viscosity": problem.viscosity,
                    "cells": cells,
                    "normalized_l1": normalized_lp(prediction, reference, 1),
                    "normalized_l2": normalized_lp(prediction, reference, 2),
                    "normalized_linf": normalized_lp(prediction, reference, np.inf),
                    "shock_location_error": abs(location - reference_location),
                    "shock_width_10_90": shock_width_10_90(
                        x, prediction, left_state, right_state, window
                    ),
                    "reference_shock_width_10_90": shock_width_10_90(
                        x, reference, left_state, right_state, window
                    ),
                    "gradient_equivalent_width": gradient_equivalent_width(
                        x, prediction, left_state, right_state, window
                    ),
                    "reference_gradient_equivalent_width": gradient_equivalent_width(
                        x, reference, left_state, right_state, window
                    ),
                    "mass_error": abs(float(np.mean(prediction) - np.mean(initial_np))),
                    "tv_excess": total_variation_excess(prediction, reference),
                    "overshoot_undershoot": overshoot_undershoot(prediction, reference),
                    "quadratic_entropy_drop": float(
                        np.mean(initial_np**2) - np.mean(prediction**2)
                    ),
                    "runtime_seconds": runtime,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
