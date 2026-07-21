"""Held-out Sod benchmark for conservative, positivity-preserving Euler subcells."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.euler_finite_volume import (
    EulerHLLCFV,
    conservative_to_primitive,
    euler_flux,
    primitive_to_conservative,
)
from cse_dkan.euler_subcell import (
    ConservativeEulerDKANSubcell,
    ConservativeEulerJumpBlendDKAN,
)
from cse_dkan.metrics import (
    gradient_equivalent_width,
    normalized_lp,
    shock_location_from_gradient,
    shock_width_10_90,
)
from cse_dkan.reference import ExactSodSolver, PrimitiveState


def safe_width(x, values, left, right, window) -> tuple[float, float]:
    try:
        return shock_width_10_90(x, values, left, right, window), 0.0
    except ValueError:
        return window[1] - window[0], 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", nargs="+", type=int, default=[100, 200, 400])
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--left", nargs=3, type=float, default=[1.0, 0.0, 1.0])
    parser.add_argument("--right", nargs=3, type=float, default=[0.125, 0.0, 0.1])
    parser.add_argument("--case-id", default="sod")
    parser.add_argument("--learned-max-cells", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument("--state-cache")
    parser.add_argument("--learned-blend-multiplier", type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dtype_name = checkpoint.get("dtype", "float32") if args.dtype == "auto" else args.dtype
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    if checkpoint.get("model_class") == "ConservativeEulerJumpBlendDKAN":
        learned = ConservativeEulerJumpBlendDKAN(**checkpoint["architecture"]).to(
            device=device, dtype=dtype
        ).eval()
        learned_method = "conservative_euler_jump_dkan_subcell"
    else:
        learned = ConservativeEulerDKANSubcell(**checkpoint["architecture"]).to(
            device=device, dtype=dtype
        ).eval()
        learned_method = "conservative_euler_dkan_subcell"
    learned.load_state_dict(checkpoint["model_state"])
    trust_multiplier = (
        checkpoint.get("inference", {}).get("learned_blend_multiplier", 1.0)
        if args.learned_blend_multiplier is None
        else args.learned_blend_multiplier
    )
    if not 0.0 <= trust_multiplier <= 1.0:
        raise ValueError("learned blend multiplier must lie in [0, 1]")
    fixed = ConservativeEulerDKANSubcell(
        hidden_width=16, correction_scale=0.0
    ).to(device=device, dtype=dtype).eval()
    coordinates = learned.uniform_coordinates(
        args.refinement, dtype=dtype, device=device
    )
    left_state = PrimitiveState(*args.left)
    right_state = PrimitiveState(*args.right)
    exact = ExactSodSolver(left=left_state, right=right_state)
    rows = []
    for cells in args.cells:
        coarse_x = (np.arange(cells) + 0.5) / cells
        initial_primitive_np = np.where(
            (coarse_x < 0.5)[:, None],
            np.asarray(args.left),
            np.asarray(args.right),
        )
        initial = primitive_to_conservative(
            torch.tensor(initial_primitive_np, dtype=dtype, device=device)
        )
        solver = EulerHLLCFV(
            n_cells=cells,
            boundary="outflow",
            reconstruction="muscl_mc",
            cfl=0.25,
        )
        cache_path = None
        if args.state_cache:
            state_digest = hashlib.sha256(
                json.dumps(
                    {
                        "left": args.left,
                        "right": args.right,
                        "time": args.final_time,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = Path(args.state_cache) / (
                f"{args.case_id}_{state_digest}_n_{cells}_{dtype_name}.npz"
            )
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            coarse_final = torch.tensor(
                cached["coarse_final"], dtype=dtype, device=device
            )
            solve_seconds = float(cached["solve_seconds"])
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            solve_started = time.perf_counter()
            with torch.no_grad():
                coarse_final = solver.solve(initial, args.final_time)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            solve_seconds = time.perf_counter() - solve_started
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    coarse_final=coarse_final.detach().cpu().numpy(),
                    solve_seconds=np.asarray(solve_seconds),
                )

        offsets = (np.arange(args.refinement) + 0.5) / args.refinement
        x = ((np.arange(cells)[:, None] + offsets[None, :]) / cells).reshape(-1)
        density, velocity, pressure = exact.sample(x, args.final_time)
        reference = np.column_stack((density, velocity, pressure))
        right_sound = np.sqrt(1.4 * right_state.pressure / right_state.density)
        pressure_ratio = exact.p_star / right_state.pressure
        right_shock_speed = right_state.velocity + right_sound * np.sqrt(
            2.4 / 2.8 * pressure_ratio + 0.4 / 2.8
        )
        reference_shock = 0.5 + right_shock_speed * args.final_time
        shock_window = (reference_shock - 0.08, reference_shock + 0.08)
        left_mask = (x > reference_shock - 0.04) & (x < reference_shock - 0.015)
        right_mask = (x > reference_shock + 0.015) & (x < reference_shock + 0.04)
        reference_left_density = float(np.mean(density[left_mask]))
        reference_right_density = float(np.mean(density[right_mask]))

        with torch.no_grad():
            learned_profile = learned(
                coarse_final,
                coordinates,
                learned_blend_multiplier=trust_multiplier,
            )
            if args.learned_max_cells is not None and cells > args.learned_max_cells:
                learned_profile = fixed(coarse_final, coordinates)
            profiles = {
                "piecewise_constant": coarse_final.unsqueeze(-2).expand(
                    -1, args.refinement, -1
                ),
                "fixed_conservative_mc_subcell": fixed(coarse_final, coordinates),
                (
                    "resolution_gated_euler_jump_dkan_subcell"
                    if args.learned_max_cells is not None
                    else learned_method
                ): learned_profile,
            }
        for method, conservative_profile in profiles.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_started = time.perf_counter()
            with torch.no_grad():
                if method == "piecewise_constant":
                    timed_profile = coarse_final.unsqueeze(-2).expand(
                        -1, args.refinement, -1
                    )
                elif method == "fixed_conservative_mc_subcell":
                    timed_profile = fixed(coarse_final, coordinates)
                elif args.learned_max_cells is not None and cells > args.learned_max_cells:
                    timed_profile = fixed(coarse_final, coordinates)
                else:
                    timed_profile = learned(
                        coarse_final,
                        coordinates,
                        learned_blend_multiplier=trust_multiplier,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - reconstruction_started
            primitive_profile = conservative_to_primitive(conservative_profile)
            primitive = primitive_profile.detach().cpu().numpy().reshape(-1, 3)
            conservative = conservative_profile.detach().cpu().numpy().reshape(-1, 3)
            location = shock_location_from_gradient(x, primitive[:, 0], shock_window)
            width, width_failure = safe_width(
                x,
                primitive[:, 0],
                reference_left_density,
                reference_right_density,
                shock_window,
            )
            predicted_left = np.mean(primitive[left_mask], axis=0)
            predicted_right = np.mean(primitive[right_mask], axis=0)
            entropy_left = np.log(predicted_left[2] / predicted_left[0] ** 1.4)
            entropy_right = np.log(predicted_right[2] / predicted_right[0] ** 1.4)
            shock_speed = (reference_shock - 0.5) / args.final_time
            states = torch.tensor(
                np.stack((predicted_left, predicted_right)),
                dtype=torch.float64,
            )
            conservative_states = primitive_to_conservative(states)
            flux_states = euler_flux(conservative_states)
            rh = (
                flux_states[1]
                - flux_states[0]
                - shock_speed * (conservative_states[1] - conservative_states[0])
            )
            reconstructed_average = conservative_profile.mean(dim=-2)
            component_l2 = [
                normalized_lp(primitive[:, component], reference[:, component], 2)
                for component in range(3)
            ]
            rows.append(
                {
                    "method": method,
                    "case_id": args.case_id,
                    "left_state": args.left,
                    "right_state": args.right,
                    "cells": cells,
                    "subcells": args.refinement,
                    "density_normalized_l2": component_l2[0],
                    "velocity_normalized_l2": component_l2[1],
                    "pressure_normalized_l2": component_l2[2],
                    "primitive_geometric_mean_l2": float(
                        np.exp(np.mean(np.log(np.maximum(component_l2, 1.0e-15))))
                    ),
                    "density_shock_location_error": abs(location - reference_shock),
                    "density_shock_width_10_90": width,
                    "density_shock_width_failure": width_failure,
                    "density_gradient_equivalent_width": gradient_equivalent_width(
                        x,
                        primitive[:, 0],
                        reference_left_density,
                        reference_right_density,
                        shock_window,
                    ),
                    "minimum_density": float(np.min(primitive[:, 0])),
                    "minimum_pressure": float(np.min(primitive[:, 2])),
                    "shock_entropy_violation": float(max(0.0, entropy_right - entropy_left)),
                    "shock_rh_residual_l2": float(torch.linalg.vector_norm(rh)),
                    "maximum_cell_average_change": float(
                        torch.max(torch.abs(reconstructed_average - coarse_final)).cpu()
                    ),
                    "learned_blend_multiplier": trust_multiplier if method not in {"piecewise_constant", "fixed_conservative_mc_subcell"} else 0.0,
                    "global_conservative_change": [
                        float(value)
                        for value in torch.abs(
                            reconstructed_average.mean(dim=0) - coarse_final.mean(dim=0)
                        ).cpu()
                    ],
                    "fv_solve_seconds": solve_seconds,
                    "reconstruction_seconds": reconstruction_seconds,
                }
            )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pretraining_generation_seconds": checkpoint["generation_seconds"],
        "pretraining_seconds": checkpoint["training_seconds"],
        "pretraining_oracle_seconds": checkpoint.get("oracle_seconds", 0.0),
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
