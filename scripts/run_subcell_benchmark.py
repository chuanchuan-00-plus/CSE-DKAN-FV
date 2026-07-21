"""Held-out continuous-profile benchmark for conservative DKAN subcells."""

from __future__ import annotations

import argparse
import json
import math
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
from cse_dkan.subcell import ConservativeDKANSubcell


def exact_initial_cell_averages(problem: PeriodicBurgersProblem, cells: int) -> np.ndarray:
    dx = (problem.x_right - problem.x_left) / cells
    edges = np.linspace(problem.x_left, problem.x_right, cells + 1)
    oscillatory = problem.amplitude / (math.pi * dx) * (
        np.cos(math.pi * (edges[1:] - problem.origin))
        - np.cos(math.pi * (edges[:-1] - problem.origin))
    )
    return problem.mean + oscillatory


def safe_width(x, values, left, right, window) -> tuple[float, float]:
    try:
        return shock_width_10_90(x, values, left, right, window), 0.0
    except ValueError:
        return window[1] - window[0], 1.0


def periodic_total_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sum(np.abs(values - np.roll(values, 1))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--reference-cells", type=int, default=4096)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[1.0])
    parser.add_argument("--viscosities", nargs="+", type=float, default=[0.0, 1.0e-3])
    parser.add_argument("--shock-sensor-threshold", type=float)
    parser.add_argument("--shock-sensor-minimum-jump", type=float, default=0.0)
    parser.add_argument("--shock-sensor-dilation", type=int, default=1)
    parser.add_argument("--initial-tv-relative-budget", type=float, default=0.0)
    parser.add_argument("--cell-tv-relative-budget", type=float)
    parser.add_argument("--disable-tv-projection", action="store_true")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument("--reference-cache")
    parser.add_argument("--adaptive-viscous-reference", action="store_true")
    parser.add_argument("--learned-correction-multiplier", type=float)
    parser.add_argument("--maximum-viscous-transition-cells", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dtype_name = checkpoint.get("dtype", "float32") if args.dtype == "auto" else args.dtype
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    learned = ConservativeDKANSubcell(**checkpoint["architecture"]).to(
        device=device, dtype=dtype
    ).eval()
    learned.load_state_dict(checkpoint["model_state"])
    trust_multiplier = (
        checkpoint.get("inference", {}).get("learned_correction_multiplier", 1.0)
        if args.learned_correction_multiplier is None
        else args.learned_correction_multiplier
    )
    if not 0.0 <= trust_multiplier <= 1.0:
        raise ValueError("learned correction multiplier must lie in [0, 1]")
    fixed = ConservativeDKANSubcell(hidden_width=16, correction_scale=0.0).to(
        device=device, dtype=dtype
    ).eval()
    coordinates = learned.uniform_coordinates(args.refinement, dtype=dtype, device=device)
    rows = []
    for amplitude in args.amplitudes:
        for viscosity, cells in (
            (viscosity, cells)
            for viscosity in args.viscosities
            for cells in args.cells
        ):
            problem = PeriodicBurgersProblem(viscosity=viscosity, amplitude=amplitude)
            reference_cells = args.reference_cells
            if args.adaptive_viscous_reference:
                if viscosity >= 1.0e-2:
                    reference_cells = min(reference_cells, 1024)
                elif viscosity >= 3.0e-3:
                    reference_cells = min(reference_cells, 2048)
            cache_path = None
            if args.reference_cache:
                cache_path = Path(args.reference_cache) / (
                    f"a_{amplitude:.17g}_nu_{viscosity:.17g}_"
                    f"n_{reference_cells}.npz"
                )
            if cache_path is not None and cache_path.exists():
                cached = np.load(cache_path)
                reference_x = cached["x"]
                reference_u = cached["u"]
            else:
                reference_x, snapshots = problem.reference(reference_cells)
                reference_u = snapshots[problem.final_time]
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, x=reference_x, u=reference_u)
            expected_shock = problem.origin + problem.mean * problem.final_time
            window = (expected_shock - 0.15, expected_shock + 0.15)
            left_state = float(np.interp(expected_shock - 0.02, reference_x, reference_u))
            right_state = float(np.interp(expected_shock + 0.02, reference_x, reference_u))
            shock_formed = problem.final_time >= problem.shock_formation_time
            dx = 2.0 / cells
            coarse_initial_np = exact_initial_cell_averages(problem, cells)
            coarse_initial = torch.tensor(
                coarse_initial_np, dtype=dtype, device=device
            )
            solver = ConservativeBurgersFV(
                n_cells=cells, limiter="tvd", viscosity=viscosity
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            solve_started = time.perf_counter()
            with torch.no_grad():
                coarse_final = solver.solve(coarse_initial, problem.final_time)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            solve_seconds = time.perf_counter() - solve_started

            offsets = (np.arange(args.refinement) + 0.5) / args.refinement
            x = (
                problem.x_left
                + (np.arange(cells)[:, None] + offsets[None, :]) * dx
            ).reshape(-1)
            reference = np.interp(x, reference_x, reference_u, period=2.0)
            learned_profile = learned(
                coarse_final,
                coordinates,
                viscosity=viscosity,
                dx=dx,
                learned_correction_multiplier=trust_multiplier,
            )
            initial_tv_budget = torch.sum(
                torch.abs(coarse_initial - torch.roll(coarse_initial, 1))
            )
            projection_tv_budget = (
                1.0 + args.initial_tv_relative_budget
            ) * initial_tv_budget
            if args.cell_tv_relative_budget is not None:
                learned_profile = learned.project_total_variation(
                    learned_profile,
                    coarse_final,
                    relative_budget=args.cell_tv_relative_budget,
                )
            elif not args.disable_tv_projection:
                learned_profile = learned.project_total_variation(
                    learned_profile, coarse_final, absolute_budget=projection_tv_budget
                )
            profiles = {
                "piecewise_constant": coarse_final.unsqueeze(-1).expand(-1, args.refinement),
                "fixed_mc_subcell": fixed(
                    coarse_final, coordinates, viscosity=viscosity, dx=dx
                ),
                "conservative_dkan_subcell": learned_profile,
            }
            if args.shock_sensor_threshold is not None:
                gated_profile = learned(
                    coarse_final,
                    coordinates,
                    viscosity=viscosity,
                    dx=dx,
                    shock_sensor_threshold=args.shock_sensor_threshold,
                    shock_sensor_minimum_jump=args.shock_sensor_minimum_jump,
                    shock_sensor_dilation=args.shock_sensor_dilation,
                    learned_correction_multiplier=trust_multiplier,
                    maximum_viscous_transition_cells=args.maximum_viscous_transition_cells,
                )
                profiles["sensor_gated_dkan_subcell"] = (
                    gated_profile
                    if args.disable_tv_projection
                    else learned.project_total_variation(
                        gated_profile, coarse_final, absolute_budget=projection_tv_budget
                    )
                )
                if args.cell_tv_relative_budget is not None:
                    profiles["sensor_gated_dkan_subcell"] = learned.project_total_variation(
                        gated_profile,
                        coarse_final,
                        relative_budget=args.cell_tv_relative_budget,
                    )
            for method, profile_tensor in profiles.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reconstruction_started = time.perf_counter()
                with torch.no_grad():
                    if method == "piecewise_constant":
                        timed_profile = coarse_final.unsqueeze(-1).expand(
                            -1, args.refinement
                        )
                    elif method == "fixed_mc_subcell":
                        timed_profile = fixed(
                            coarse_final, coordinates, viscosity=viscosity, dx=dx
                        )
                    elif method == "conservative_dkan_subcell":
                        timed_profile = learned(
                            coarse_final,
                            coordinates,
                            viscosity=viscosity,
                            dx=dx,
                            learned_correction_multiplier=trust_multiplier,
                            maximum_viscous_transition_cells=args.maximum_viscous_transition_cells,
                        )
                        if args.cell_tv_relative_budget is not None:
                            timed_profile = learned.project_total_variation(
                                timed_profile,
                                coarse_final,
                                relative_budget=args.cell_tv_relative_budget,
                            )
                        elif not args.disable_tv_projection:
                            timed_profile = learned.project_total_variation(
                                timed_profile,
                                coarse_final,
                                absolute_budget=projection_tv_budget,
                            )
                    else:
                        timed_profile = learned(
                            coarse_final,
                            coordinates,
                            viscosity=viscosity,
                            dx=dx,
                            shock_sensor_threshold=args.shock_sensor_threshold,
                            shock_sensor_minimum_jump=args.shock_sensor_minimum_jump,
                            shock_sensor_dilation=args.shock_sensor_dilation,
                            learned_correction_multiplier=trust_multiplier,
                            maximum_viscous_transition_cells=args.maximum_viscous_transition_cells,
                        )
                        if args.cell_tv_relative_budget is not None:
                            timed_profile = learned.project_total_variation(
                                timed_profile,
                                coarse_final,
                                relative_budget=args.cell_tv_relative_budget,
                            )
                        elif not args.disable_tv_projection:
                            timed_profile = learned.project_total_variation(
                                timed_profile,
                                coarse_final,
                                absolute_budget=projection_tv_budget,
                            )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reconstruction_seconds = time.perf_counter() - reconstruction_started
                profile = profile_tensor.detach().cpu().numpy().reshape(-1)
                profile_tv = periodic_total_variation(profile)
                initial_tv = float(initial_tv_budget.detach().cpu())
                width, width_failure = safe_width(
                    x, profile, left_state, right_state, window
                )
                location = shock_location_from_gradient(x, profile, window)
                reference_location = shock_location_from_gradient(x, reference, window)
                reconstructed_averages = profile.reshape(cells, args.refinement).mean(axis=1)
                rows.append(
                    {
                        "method": method,
                        "viscosity": viscosity,
                        "amplitude": amplitude,
                        "shock_formed": shock_formed,
                        "shock_formation_time": problem.shock_formation_time,
                        "cells": cells,
                        "subcells": args.refinement,
                        "reference_cells": reference_cells,
                        "tv_projection_relative_budget": args.initial_tv_relative_budget,
                        "cell_tv_projection_relative_budget": args.cell_tv_relative_budget,
                        "tv_projection_enabled": not args.disable_tv_projection,
                        "shock_sensor_threshold": args.shock_sensor_threshold if method == "sensor_gated_dkan_subcell" else None,
                        "shock_sensor_minimum_jump": args.shock_sensor_minimum_jump if method == "sensor_gated_dkan_subcell" else 0.0,
                        "shock_sensor_dilation": args.shock_sensor_dilation if method == "sensor_gated_dkan_subcell" else 0,
                        "learned_correction_multiplier": trust_multiplier if method in {"conservative_dkan_subcell", "sensor_gated_dkan_subcell"} else 0.0,
                        "maximum_viscous_transition_cells": args.maximum_viscous_transition_cells if method == "sensor_gated_dkan_subcell" else None,
                        "tv_projection_absolute_budget": float(projection_tv_budget.detach().cpu()) if method in {"conservative_dkan_subcell", "sensor_gated_dkan_subcell"} else 0.0,
                        "periodic_total_variation": profile_tv,
                        "initial_periodic_total_variation": initial_tv,
                        "initial_tvd_violation": max(
                            0.0, (profile_tv - initial_tv) / max(initial_tv, 1.0e-14)
                        ),
                        "normalized_l1": normalized_lp(profile, reference, 1),
                        "normalized_l2": normalized_lp(profile, reference, 2),
                        "normalized_linf": normalized_lp(profile, reference, np.inf),
                        "shock_location_error": abs(location - reference_location),
                        "shock_width_10_90": width,
                        "shock_width_failure": width_failure,
                        "gradient_equivalent_width": gradient_equivalent_width(
                            x, profile, left_state, right_state, window
                        ),
                        "tv_excess": total_variation_excess(profile, reference),
                        "overshoot_undershoot": overshoot_undershoot(profile, reference),
                        "maximum_cell_average_change": float(
                            np.max(
                                np.abs(
                                    reconstructed_averages
                                    - coarse_final.detach().cpu().numpy()
                                )
                            )
                        ),
                        "global_mass_error": abs(
                            float(np.mean(profile) - np.mean(coarse_initial_np))
                        ),
                        "fv_solve_seconds": solve_seconds,
                        "reconstruction_seconds": reconstruction_seconds,
                    }
                )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pretraining_generation_seconds": checkpoint["generation_seconds"],
        "pretraining_seconds": checkpoint["training_seconds"],
        "dtype": dtype_name,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
