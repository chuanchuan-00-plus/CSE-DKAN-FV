"""Held-out Burgers benchmark for fixed and DKAN-corrected FV-WENO."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.finite_volume import ConservativeBurgersFV, DKANWENO5Reconstruction
from cse_dkan.metrics import (
    gradient_equivalent_width,
    normalized_lp,
    overshoot_undershoot,
    shock_location_from_gradient,
    shock_width_10_90,
    total_variation_excess,
)


def load_reconstruction(path: Path, device: torch.device) -> tuple[DKANWENO5Reconstruction, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload["architecture"]
    model = DKANWENO5Reconstruction(**architecture).to(device).eval()
    model.load_state_dict(payload["model_state"])
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--reference-cells", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learned, checkpoint = load_reconstruction(Path(args.checkpoint), device)
    fixed = DKANWENO5Reconstruction(hidden_width=8, correction_scale=0.0).to(device).eval()
    rows: list[dict[str, float | int | str]] = []
    for viscosity in (0.0, 1.0e-3):
        problem = PeriodicBurgersProblem(viscosity=viscosity)
        reference_x, snapshots = problem.reference(args.reference_cells)
        reference_u = snapshots[problem.final_time]
        expected_shock = problem.origin + problem.mean * problem.final_time
        window = (expected_shock - 0.15, expected_shock + 0.15)
        left_state = float(np.interp(expected_shock - 0.02, reference_x, reference_u))
        right_state = float(np.interp(expected_shock + 0.02, reference_x, reference_u))
        for cells in args.cells:
            x = problem.x_left + (np.arange(cells) + 0.5) * (
                problem.x_right - problem.x_left
            ) / cells
            initial_np = problem.initial_numpy(x)
            reference = np.interp(
                x, reference_x, reference_u, period=problem.x_right - problem.x_left
            )
            reference_location = shock_location_from_gradient(x, reference, window)
            for method, reconstruction in (
                ("tvd_mc_fv", "tvd"),
                ("bounded_weno5z_fv", fixed),
                ("cse_fv_dkan_weno", learned),
            ):
                initial = torch.tensor(initial_np, dtype=torch.float32, device=device)
                solver = ConservativeBurgersFV(
                    n_cells=cells,
                    cfl=0.20 if isinstance(reconstruction, torch.nn.Module) else 0.35,
                    limiter=reconstruction,
                    viscosity=viscosity,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.no_grad():
                    prediction = solver.solve(initial, problem.final_time)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime = time.perf_counter() - started
                prediction_np = prediction.detach().cpu().numpy()
                location = shock_location_from_gradient(x, prediction_np, window)
                rows.append(
                    {
                        "method": method,
                        "viscosity": viscosity,
                        "cells": cells,
                        "normalized_l1": normalized_lp(prediction_np, reference, 1),
                        "normalized_l2": normalized_lp(prediction_np, reference, 2),
                        "normalized_linf": normalized_lp(prediction_np, reference, np.inf),
                        "shock_location_error": abs(location - reference_location),
                        "shock_width_10_90": shock_width_10_90(
                            x, prediction_np, left_state, right_state, window
                        ),
                        "reference_shock_width_10_90": shock_width_10_90(
                            x, reference, left_state, right_state, window
                        ),
                        "gradient_equivalent_width": gradient_equivalent_width(
                            x, prediction_np, left_state, right_state, window
                        ),
                        "reference_gradient_equivalent_width": gradient_equivalent_width(
                            x, reference, left_state, right_state, window
                        ),
                        "mass_error": abs(float(np.mean(prediction_np) - np.mean(initial_np))),
                        "tv_excess": total_variation_excess(prediction_np, reference),
                        "overshoot_undershoot": overshoot_undershoot(
                            prediction_np, reference
                        ),
                        "runtime_seconds": runtime,
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pretraining_generation_seconds": checkpoint["generation_seconds"],
        "pretraining_seconds": checkpoint["training_seconds"],
        "rows": rows,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
