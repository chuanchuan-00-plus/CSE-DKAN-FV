"""Broad parametric Riemann benchmark for FNO, finite volume, and DKAN subcells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    density_tv_trust_projection,
)
from cse_dkan.fno import FNO1d
from cse_dkan.metrics import normalized_lp, shock_width_10_90, total_variation_excess
from cse_dkan.reference import ExactSodSolver, PrimitiveState


def physical_entropy(primitive: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    return np.log(primitive[..., 2]) - gamma * np.log(primitive[..., 0])


def shock_speeds(exact: ExactSodSolver) -> list[tuple[str, float]]:
    speeds = []
    gamma = exact.gamma
    if exact.p_star > exact.left.pressure:
        sound = math.sqrt(gamma * exact.left.pressure / exact.left.density)
        speed = exact.left.velocity - sound * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * exact.p_star / exact.left.pressure
            + (gamma - 1.0) / (2.0 * gamma)
        )
        speeds.append(("left", speed))
    if exact.p_star > exact.right.pressure:
        sound = math.sqrt(gamma * exact.right.pressure / exact.right.density)
        speed = exact.right.velocity + sound * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * exact.p_star / exact.right.pressure
            + (gamma - 1.0) / (2.0 * gamma)
        )
        speeds.append(("right", speed))
    return speeds


def discontinuity_metrics(
    x: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    exact: ExactSodSolver,
    final_time: float,
) -> dict:
    dx = float(x[1] - x[0])
    widths = []
    entropy_violations = []
    rh_residuals = []
    for orientation, speed in shock_speeds(exact):
        location = exact.discontinuity + speed * final_time
        if not 8.0 * dx < location < 1.0 - 8.0 * dx:
            continue
        left_mask = (x > location - 7.0 * dx) & (x < location - 3.0 * dx)
        right_mask = (x > location + 3.0 * dx) & (x < location + 7.0 * dx)
        if not np.any(left_mask) or not np.any(right_mask):
            continue
        reference_left = float(np.mean(reference[left_mask, 0]))
        reference_right = float(np.mean(reference[right_mask, 0]))
        try:
            widths.append(
                shock_width_10_90(
                    x,
                    prediction[:, 0],
                    reference_left,
                    reference_right,
                    (max(0.0, location - 0.08), min(1.0, location + 0.08)),
                )
            )
        except ValueError:
            widths.append(0.16)
        predicted_left = np.mean(prediction[left_mask], axis=0)
        predicted_right = np.mean(prediction[right_mask], axis=0)
        entropy_left = float(physical_entropy(predicted_left))
        entropy_right = float(physical_entropy(predicted_right))
        entropy_violations.append(
            max(0.0, entropy_left - entropy_right)
            if orientation == "left"
            else max(0.0, entropy_right - entropy_left)
        )
        states = torch.tensor(
            np.stack((predicted_left, predicted_right)), dtype=torch.float64
        )
        conservative = primitive_to_conservative(states)
        flux = euler_flux(conservative)
        residual = flux[1] - flux[0] - speed * (conservative[1] - conservative[0])
        scale = (
            torch.linalg.vector_norm(flux[1] - flux[0])
            + abs(speed) * torch.linalg.vector_norm(conservative[1] - conservative[0])
            + 1.0e-12
        )
        rh_residuals.append(float(torch.linalg.vector_norm(residual) / scale))
    return {
        "mean_shock_width_10_90": float(np.mean(widths)) if widths else None,
        "maximum_shock_entropy_violation": max(entropy_violations, default=0.0),
        "maximum_normalized_rh_residual": max(rh_residuals, default=0.0),
        "evaluated_shock_count": len(widths),
    }


def evaluate(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    x: np.ndarray,
    exact: ExactSodSolver,
    final_time: float,
    online_seconds: float,
    model_seed: int | None = None,
    maximum_cell_average_change: float | None = None,
) -> dict:
    component_l2 = [
        normalized_lp(prediction[:, component], reference[:, component], 2)
        for component in range(3)
    ]
    predicted_conservative = primitive_to_conservative(
        torch.tensor(prediction, dtype=torch.float64)
    ).numpy()
    reference_conservative = primitive_to_conservative(
        torch.tensor(reference, dtype=torch.float64)
    ).numpy()
    conservation = np.abs(
        np.mean(predicted_conservative, axis=0)
        - np.mean(reference_conservative, axis=0)
    ) / np.maximum(np.mean(np.abs(reference_conservative), axis=0), 1.0e-12)
    return {
        "method": method,
        "model_seed": model_seed,
        "density_normalized_l2": component_l2[0],
        "velocity_normalized_l2": component_l2[1],
        "pressure_normalized_l2": component_l2[2],
        "primitive_geometric_mean_l2": float(
            np.exp(np.mean(np.log(np.maximum(component_l2, 1.0e-15))))
        ),
        "maximum_relative_conservation_error": float(np.max(conservation)),
        "relative_conservation_errors": [float(value) for value in conservation],
        "minimum_density": float(np.min(prediction[:, 0])),
        "minimum_pressure": float(np.min(prediction[:, 2])),
        "density_tv_excess": total_variation_excess(
            prediction[:, 0], reference[:, 0]
        ),
        "online_seconds": online_seconds,
        "maximum_cell_average_change": maximum_cell_average_change,
        **discontinuity_metrics(x, prediction, reference, exact, final_time),
    }


def cached_fv_state(
    initial: torch.Tensor,
    final_time: float,
    cells: int,
    cache_root: Path | None,
) -> tuple[torch.Tensor, float]:
    digest = hashlib.sha256(
        initial.detach().cpu().numpy().tobytes()
        + np.asarray([final_time, cells], dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    path = cache_root / f"state_{digest}_n{cells}.npz" if cache_root else None
    if path is not None and path.exists():
        cached = np.load(path)
        return torch.tensor(cached["state"], dtype=torch.float64, device=initial.device), float(
            cached["seconds"]
        )
    solver = EulerHLLCFV(
        n_cells=cells,
        boundary="outflow",
        reconstruction="muscl_mc",
        cfl=0.25,
    )
    if initial.device.type == "cuda":
        torch.cuda.synchronize(initial.device)
    started = time.perf_counter()
    with torch.no_grad():
        state = solver.solve(initial, final_time)
    if initial.device.type == "cuda":
        torch.cuda.synchronize(initial.device)
    seconds = time.perf_counter() - started
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, state=state.detach().cpu().numpy(), seconds=seconds)
    return state, seconds


def load_fno(path: Path, device: torch.device) -> tuple[FNO1d, dict, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FNO1d(**checkpoint["architecture"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    normalization = {
        name: value.to(device) for name, value in checkpoint["normalization"].items()
    }
    return (
        model,
        normalization,
        int(checkpoint["training"]["seed"]),
        float(checkpoint["generation_seconds"] + checkpoint["training_seconds"]),
    )


def load_hybrid(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConservativeEulerJumpBlendDKAN(**checkpoint["architecture"]).to(
        device=device, dtype=torch.float64
    ).eval()
    model.load_state_dict(checkpoint["model_state"])
    seed = int(checkpoint["training"]["seed"])
    family = str(checkpoint["training"].get("case_family", "unknown"))
    pretraining = float(
        checkpoint["generation_seconds"]
        + checkpoint.get("oracle_seconds", 0.0)
        + checkpoint["training_seconds"]
    )
    return model, seed, family, pretraining


def fno_input(case: dict, points: int, final_time: float, device: torch.device) -> torch.Tensor:
    x = (np.arange(points, dtype=np.float32) + 0.5) / points
    left = np.asarray(case["left"], dtype=np.float32)
    right = np.asarray(case["right"], dtype=np.float32)
    primitive = np.where((x < 0.5)[:, None], left, right)
    values = np.column_stack(
        (
            np.log(primitive[:, 0]),
            primitive[:, 1],
            np.log(primitive[:, 2]),
            2.0 * x - 1.0,
            np.full_like(x, final_time),
        )
    )
    return torch.tensor(values[None], dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fno-checkpoints", nargs="+", required=True)
    parser.add_argument("--hybrid-checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-cache")
    parser.add_argument("--tv-trust-relative-budgets", nargs="*", type=float, default=[])
    args = parser.parse_args()
    with open(args.protocol, encoding="utf-8") as handle:
        protocol = json.load(handle)
    configuration = protocol["benchmark"]
    points = int(configuration["evaluation_points"])
    coarse_cells = int(configuration["coarse_cells"])
    refinement = points // coarse_cells
    fine_cells = int(configuration["fine_hllc_cells"])
    if refinement * coarse_cells != points:
        raise ValueError("evaluation_points must be divisible by coarse_cells")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fno_models = [load_fno(Path(path), device) for path in args.fno_checkpoints]
    hybrid_models = [load_hybrid(Path(path), device) for path in args.hybrid_checkpoints]
    fixed = ConservativeEulerDKANSubcell(
        hidden_width=16, correction_scale=0.0
    ).to(device=device, dtype=torch.float64).eval()
    coordinates = fixed.uniform_coordinates(
        refinement, dtype=torch.float64, device=device
    )
    x = (np.arange(points, dtype=np.float64) + 0.5) / points
    rows = []
    cache_root = Path(args.state_cache) if args.state_cache else None
    for case in protocol["cases"]:
        final_time = float(case["time"])
        left = PrimitiveState(*case["left"])
        right = PrimitiveState(*case["right"])
        exact = ExactSodSolver(left=left, right=right)
        reference = np.column_stack(exact.sample(x, final_time))
        coarse_x = (np.arange(coarse_cells) + 0.5) / coarse_cells
        coarse_primitive = np.where(
            (coarse_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        coarse_initial = primitive_to_conservative(
            torch.tensor(coarse_primitive, dtype=torch.float64, device=device)
        )
        coarse_state, coarse_seconds = cached_fv_state(
            coarse_initial, final_time, coarse_cells, cache_root
        )
        fine_x = (np.arange(fine_cells) + 0.5) / fine_cells
        fine_primitive = np.where(
            (fine_x < 0.5)[:, None], np.asarray(case["left"]), np.asarray(case["right"])
        )
        fine_initial = primitive_to_conservative(
            torch.tensor(fine_primitive, dtype=torch.float64, device=device)
        )
        fine_state, fine_seconds = cached_fv_state(
            fine_initial, final_time, fine_cells, cache_root
        )
        fine_profile = conservative_to_primitive(fine_state).detach().cpu().numpy()
        if fine_cells != points:
            fine_profile = np.column_stack(
                [np.interp(x, fine_x, fine_profile[:, component]) for component in range(3)]
            )
        with torch.no_grad():
            fixed_conservative = fixed(coarse_state, coordinates)
        fixed_profile = conservative_to_primitive(fixed_conservative).detach().cpu().numpy().reshape(-1, 3)
        piecewise_profile = conservative_to_primitive(
            coarse_state.unsqueeze(-2).expand(-1, refinement, -1)
        ).detach().cpu().numpy().reshape(-1, 3)
        for method, profile, seconds in (
            ("hllc_piecewise_constant", piecewise_profile, coarse_seconds),
            ("hllc_fixed_mc_subcell", fixed_profile, coarse_seconds),
            ("hllc_muscl_fine", fine_profile, fine_seconds),
        ):
            rows.append(
                {
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "time": final_time,
                    **evaluate(method, profile, reference, x, exact, final_time, seconds),
                }
            )
        for model, seed, family, _ in hybrid_models:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                conservative = model(
                    coarse_state,
                    coordinates,
                    learned_blend_multiplier=configuration["hybrid_blend_multiplier"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - started
            change = float(
                torch.max(torch.abs(conservative.mean(dim=-2) - coarse_state)).cpu()
            )
            profile = conservative_to_primitive(conservative).detach().cpu().numpy().reshape(-1, 3)
            rows.append(
                {
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "time": final_time,
                    **evaluate(
                        f"physics_gated_dkan_fv_{family}",
                        profile,
                        reference,
                        x,
                        exact,
                        final_time,
                        coarse_seconds + reconstruction_seconds,
                        model_seed=seed,
                        maximum_cell_average_change=change,
                    ),
                }
            )
            for relative_budget in args.tv_trust_relative_budgets:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                projection_started = time.perf_counter()
                with torch.no_grad():
                    trusted, multiplier = density_tv_trust_projection(
                        fixed_conservative,
                        conservative,
                        relative_budget=relative_budget,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                projection_seconds = time.perf_counter() - projection_started
                trusted_change = float(
                    torch.max(torch.abs(trusted.mean(dim=-2) - coarse_state)).cpu()
                )
                trusted_profile = (
                    conservative_to_primitive(trusted)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)
                )
                row = {
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "time": final_time,
                    **evaluate(
                        f"physics_gated_dkan_fv_{family}_tv_trust_{relative_budget:g}",
                        trusted_profile,
                        reference,
                        x,
                        exact,
                        final_time,
                        coarse_seconds + reconstruction_seconds + projection_seconds,
                        model_seed=seed,
                        maximum_cell_average_change=trusted_change,
                    ),
                }
                row["tv_trust_multiplier"] = float(multiplier.cpu())
                row["tv_trust_relative_budget"] = relative_budget
                rows.append(row)
        for model, normalization, seed, _ in fno_models:
            values = fno_input(case, points, final_time, device)
            normalized = (values - normalization["input_mean"]) / normalization["input_std"]
            with torch.no_grad():
                _ = model(normalized)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                transformed = (
                    model(normalized) * normalization["target_std"]
                    + normalization["target_mean"]
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            transformed = transformed[0].detach().cpu().numpy()
            profile = np.column_stack(
                (np.exp(transformed[:, 0]), transformed[:, 1], np.exp(transformed[:, 2]))
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "time": final_time,
                    **evaluate(
                        "fno_operator",
                        profile,
                        reference,
                        x,
                        exact,
                        final_time,
                        seconds,
                        model_seed=seed,
                    ),
                }
            )
    result = {
        "protocol": protocol["protocol"],
        "configuration": configuration,
        "fno_pretraining": [
            {"seed": seed, "seconds": seconds}
            for _, _, seed, seconds in fno_models
        ],
        "hybrid_pretraining": [
            {"seed": seed, "case_family": family, "seconds": seconds}
            for _, seed, family, seconds in hybrid_models
        ],
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"row_count": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
