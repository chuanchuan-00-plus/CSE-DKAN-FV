"""Benchmark the locally conservative Burgers/Godunov and Euler/HLLC backbones."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.euler_finite_volume import EulerHLLCFV, conservative_to_primitive, primitive_to_conservative
from cse_dkan.finite_volume import ConservativeBurgersFV
from cse_dkan.metrics import normalized_lp, shock_location_from_gradient, shock_width_10_90
from cse_dkan.reference import ExactSodSolver


def burgers_rows(cell_counts: list[int]) -> list[dict[str, float | str]]:
    problem = PeriodicBurgersProblem()
    reference_x, snapshots = problem.reference(4096)
    reference_u = snapshots[problem.final_time]
    shock = problem.origin + problem.mean * problem.final_time
    rows = []
    for limiter in ("first_order", "tvd"):
        for cells in cell_counts:
            x = problem.x_left + (np.arange(cells) + 0.5) * (problem.x_right - problem.x_left) / cells
            initial = torch.tensor(problem.initial_numpy(x), dtype=torch.float64)
            solver = ConservativeBurgersFV(n_cells=cells, limiter=limiter)
            started = time.perf_counter()
            prediction = solver.solve(initial, problem.final_time).detach().numpy()
            elapsed = time.perf_counter() - started
            reference = np.interp(x, reference_x, reference_u, period=problem.x_right - problem.x_left)
            left_state = float(np.interp(shock - 0.02, reference_x, reference_u))
            right_state = float(np.interp(shock + 0.02, reference_x, reference_u))
            width = shock_width_10_90(
                x, prediction, left_state, right_state, (shock - 0.15, shock + 0.15)
            )
            location = shock_location_from_gradient(x, prediction, (shock - 0.15, shock + 0.15))
            reference_location = shock_location_from_gradient(x, reference, (shock - 0.15, shock + 0.15))
            rows.append(
                {
                    "problem": "inviscid_burgers_sine",
                    "method": f"godunov_{limiter}",
                    "cells": cells,
                    "normalized_l2": normalized_lp(prediction, reference, 2),
                    "shock_width_10_90": width,
                    "shock_location_error": abs(location - reference_location),
                    "mass_error": abs(float(np.mean(prediction) - np.mean(initial.numpy()))),
                    "quadratic_entropy_drop": float(torch.mean(initial**2) - np.mean(prediction**2)),
                    "runtime_seconds": elapsed,
                }
            )
    return rows


def sod_rows(cell_counts: list[int]) -> list[dict[str, float | str]]:
    exact = ExactSodSolver()
    final_time = 0.2
    rows = []
    for reconstruction in ("first_order", "muscl_mc"):
        for cells in cell_counts:
            x = (np.arange(cells) + 0.5) / cells
            primitive = np.where(
                (x < 0.5)[:, None], np.array([1.0, 0.0, 1.0]), np.array([0.125, 0.0, 0.1])
            )
            initial = primitive_to_conservative(torch.tensor(primitive, dtype=torch.float64))
            solver = EulerHLLCFV(
                n_cells=cells,
                boundary="outflow",
                reconstruction=reconstruction,
                cfl=0.25 if reconstruction == "muscl_mc" else 0.35,
            )
            started = time.perf_counter()
            conservative = solver.solve(initial, final_time)
            elapsed = time.perf_counter() - started
            prediction = conservative_to_primitive(conservative).detach().numpy()
            rho, velocity, pressure = exact.sample(x, final_time)
            reference = np.column_stack((rho, velocity, pressure))
            shock_location = shock_location_from_gradient(x, prediction[:, 0], (0.75, 0.95))
            reference_location = shock_location_from_gradient(x, reference[:, 0], (0.75, 0.95))
            left_state = float(np.mean(reference[(x > reference_location - 0.03) & (x < reference_location - 0.01), 0]))
            right_state = float(np.mean(reference[(x > reference_location + 0.01) & (x < reference_location + 0.03), 0]))
            totals = conservative.mean(dim=0).detach().numpy()
            initial_totals = initial.mean(dim=0).numpy()
            rows.append(
                {
                    "problem": "euler_sod",
                    "method": f"hllc_{reconstruction}",
                "cells": cells,
                "density_normalized_l1": normalized_lp(prediction[:, 0], reference[:, 0], 1),
                "density_normalized_l2": normalized_lp(prediction[:, 0], reference[:, 0], 2),
                "velocity_normalized_l2": normalized_lp(prediction[:, 1], reference[:, 1], 2),
                "pressure_normalized_l2": normalized_lp(prediction[:, 2], reference[:, 2], 2),
                "density_shock_width_10_90": shock_width_10_90(
                    x, prediction[:, 0], left_state, right_state, (0.75, 0.95)
                ),
                "shock_location_error": abs(shock_location - reference_location),
                "mass_balance_error": abs(float(totals[0] - initial_totals[0])),
                "momentum_boundary_balance_error": abs(
                    float((totals[1] - initial_totals[1]) - 0.9 * final_time)
                ),
                "energy_balance_error": abs(float(totals[2] - initial_totals[2])),
                "minimum_density": float(np.min(prediction[:, 0])),
                "minimum_pressure": float(np.min(prediction[:, 2])),
                    "runtime_seconds": elapsed,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    rows = burgers_rows([64, 128, 256, 512]) + sod_rows([100, 200, 400, 800])
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
