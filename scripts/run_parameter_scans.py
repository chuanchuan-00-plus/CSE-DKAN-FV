"""Development parameter scans for viscous Burgers and Euler pressure ratio."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.euler_finite_volume import (
    EulerHLLCFV,
    conservative_to_primitive,
    primitive_to_conservative,
)
from cse_dkan.finite_volume import ConservativeBurgersFV
from cse_dkan.metrics import normalized_lp, shock_location_from_gradient, shock_width_10_90
from cse_dkan.reference import ExactSodSolver, PrimitiveState


def viscous_burgers_scan(fv_cells: int) -> list[dict[str, float | int | str]]:
    reference_cells = {
        1.0e-2: 1024,
        3.0e-3: 2048,
        1.0e-3: 4096,
        3.0e-4: 4096,
        1.0e-4: 8192,
    }
    rows = []
    for viscosity, fine_cells in reference_cells.items():
        problem = PeriodicBurgersProblem(viscosity=viscosity)
        reference_x, snapshots = problem.reference(fine_cells)
        reference_u = snapshots[problem.final_time]
        coarse_x, coarse_snapshots = problem.reference(fine_cells // 2)
        coarse_reference = coarse_snapshots[problem.final_time]
        restricted = np.interp(coarse_x, reference_x, reference_u, period=2.0)
        reference_convergence_l2 = normalized_lp(coarse_reference, restricted, 2)
        x = problem.x_left + (np.arange(fv_cells) + 0.5) * 2.0 / fv_cells
        initial_np = problem.initial_numpy(x)
        reference = np.interp(x, reference_x, reference_u, period=2.0)
        expected_shock = problem.origin + problem.mean * problem.final_time
        window = (expected_shock - 0.15, expected_shock + 0.15)
        left_state = float(np.interp(expected_shock - 0.02, reference_x, reference_u))
        right_state = float(np.interp(expected_shock + 0.02, reference_x, reference_u))
        reference_width = shock_width_10_90(
            reference_x, reference_u, left_state, right_state, window
        )
        for limiter in ("first_order", "tvd"):
            initial = torch.tensor(initial_np, dtype=torch.float64)
            solver = ConservativeBurgersFV(
                n_cells=fv_cells, limiter=limiter, viscosity=viscosity
            )
            started = time.perf_counter()
            prediction = solver.solve(initial, problem.final_time).detach().numpy()
            runtime = time.perf_counter() - started
            width = shock_width_10_90(x, prediction, left_state, right_state, window)
            rows.append(
                {
                    "problem": "viscous_burgers_sine",
                    "method": f"godunov_{limiter}",
                    "viscosity": viscosity,
                    "cells": fv_cells,
                    "reference_cells": fine_cells,
                    "reference_half_to_fine_l2": reference_convergence_l2,
                    "normalized_l2": normalized_lp(prediction, reference, 2),
                    "shock_width_10_90": width,
                    "reference_shock_width_10_90": reference_width,
                    "shock_width_absolute_error": abs(width - reference_width),
                    "mass_error": abs(float(np.mean(prediction) - np.mean(initial_np))),
                    "minimum": float(np.min(prediction)),
                    "maximum": float(np.max(prediction)),
                    "runtime_seconds": runtime,
                }
            )
    return rows


def euler_pressure_scan(cells: int, final_time: float = 0.1) -> list[dict[str, float | int | str]]:
    rows = []
    x = (np.arange(cells) + 0.5) / cells
    for pressure_ratio in (2.0, 10.0, 100.0):
        left = PrimitiveState(1.0, 0.0, 0.1 * pressure_ratio)
        right = PrimitiveState(0.125, 0.0, 0.1)
        exact = ExactSodSolver(left=left, right=right)
        rho, velocity, pressure = exact.sample(x, final_time)
        reference = np.column_stack((rho, velocity, pressure))
        sound_right = np.sqrt(exact.gamma * right.pressure / right.density)
        if exact.p_star <= right.pressure:
            raise ValueError("pressure-ratio scan expected a right-going shock")
        shock_speed = right.velocity + sound_right * np.sqrt(
            (exact.gamma + 1.0) / (2.0 * exact.gamma) * exact.p_star / right.pressure
            + (exact.gamma - 1.0) / (2.0 * exact.gamma)
        )
        exact_shock = exact.discontinuity + shock_speed * final_time
        shock_window = (max(0.5, exact_shock - 0.10), min(0.99, exact_shock + 0.10))
        primitive = np.where(
            (x < 0.5)[:, None],
            np.array([left.density, left.velocity, left.pressure]),
            np.array([right.density, right.velocity, right.pressure]),
        )
        initial = primitive_to_conservative(torch.tensor(primitive, dtype=torch.float64))
        for reconstruction in ("first_order", "muscl_mc"):
            solver = EulerHLLCFV(
                n_cells=cells,
                boundary="outflow",
                reconstruction=reconstruction,
                cfl=0.25 if reconstruction == "muscl_mc" else 0.35,
            )
            started = time.perf_counter()
            conservative = solver.solve(initial, final_time)
            runtime = time.perf_counter() - started
            prediction = conservative_to_primitive(conservative).detach().numpy()
            # Pressure is continuous across the contact, so its gradient
            # isolates the shock more reliably than density in weak cases.
            predicted_shock = shock_location_from_gradient(
                x, prediction[:, 2], shock_window
            )
            try:
                density_shock_width = shock_width_10_90(
                    x,
                    prediction[:, 0],
                    exact.right_star_state.density,
                    right.density,
                    shock_window,
                )
                width_failure = 0.0
            except ValueError:
                density_shock_width = shock_window[1] - shock_window[0]
                width_failure = 1.0
            totals = conservative.mean(dim=0).detach().numpy()
            initial_totals = initial.mean(dim=0).numpy()
            rows.append(
                {
                    "problem": "euler_pressure_ratio_scan",
                    "method": f"hllc_{reconstruction}",
                    "pressure_ratio": pressure_ratio,
                    "cells": cells,
                    "final_time": final_time,
                    "density_normalized_l2": normalized_lp(
                        prediction[:, 0], reference[:, 0], 2
                    ),
                    "velocity_normalized_l2": normalized_lp(
                        prediction[:, 1], reference[:, 1], 2
                    ),
                    "pressure_normalized_l2": normalized_lp(
                        prediction[:, 2], reference[:, 2], 2
                    ),
                    "density_shock_width_10_90": density_shock_width,
                    "density_shock_width_failure": width_failure,
                    "shock_location_error": abs(predicted_shock - exact_shock),
                    "mass_balance_error": abs(float(totals[0] - initial_totals[0])),
                    "momentum_boundary_balance_error": abs(
                        float(
                            (totals[1] - initial_totals[1])
                            - (left.pressure - right.pressure) * final_time
                        )
                    ),
                    "energy_balance_error": abs(float(totals[2] - initial_totals[2])),
                    "minimum_density": float(np.min(prediction[:, 0])),
                    "minimum_pressure": float(np.min(prediction[:, 2])),
                    "runtime_seconds": runtime,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--burgers-cells", type=int, default=512)
    parser.add_argument("--euler-cells", type=int, default=400)
    args = parser.parse_args()
    result = {
        "viscous_burgers": viscous_burgers_scan(args.burgers_cells),
        "euler_pressure_ratio": euler_pressure_scan(args.euler_cells),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
