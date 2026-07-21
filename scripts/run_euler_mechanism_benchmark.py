"""Frozen mechanism-diverse Euler benchmark for six deployed model families.

The script evaluates already-trained checkpoints.  It never updates a model on
an audited case.  Single-interface Riemann cases use the exact ideal-gas
solution; smooth entropy advection is analytic; Shu--Osher and the interacting
blast use a fine HLLC-MUSCL reference that is audited separately for grid
convergence.
"""

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
    primitive_to_conservative,
)
from cse_dkan.euler_subcell import (
    ConservativeEulerDKANSubcell,
    ConservativeEulerJumpBlendDKAN,
    density_tv_trust_projection,
)
from cse_dkan.fno import FNO1d
from cse_dkan.reference import ExactSodSolver, PrimitiveState


BASE_METHODS = (
    "fno_operator",
    "hllc_piecewise_constant",
    "hllc_fixed_mc_subcell",
    "physics_gated_dkan_fv_sod_neighbourhood",
    "physics_gated_dkan_fv_engineering_tv_trust_0.02",
    "hllc_muscl_fine",
)


def load_protocol(path: Path) -> dict:
    """Load a protocol and an optional one-level immutable amendment."""
    child = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in child:
        return child
    base_path = (path.parent / child["extends"]).resolve()
    base = load_protocol(base_path)
    merged = dict(base)
    merged.update({key: value for key, value in child.items() if key not in {"extends", "benchmark_overrides"}})
    merged["benchmark"] = dict(base["benchmark"])
    merged["benchmark"].update(child.get("benchmark_overrides", {}))
    if child.get("case_overrides"):
        overrides = child["case_overrides"]
        merged["cases"] = [
            {**case, **overrides.get(case["id"], {})} for case in base["cases"]
        ]
    merged["base_protocol"] = str(base_path)
    return merged


def initial_primitive(case: dict, x: np.ndarray) -> np.ndarray:
    kind = case["kind"]
    if kind == "riemann":
        left = np.asarray(case["left"], dtype=np.float64)
        right = np.asarray(case["right"], dtype=np.float64)
        return np.where((x < float(case["discontinuity"]))[:, None], left, right)
    if kind == "shu_osher":
        density = float(case["density_base"]) + float(case["density_amplitude"]) * np.sin(
            float(case["density_wavenumber"]) * x + float(case["phase"])
        )
        right = np.column_stack(
            (
                density,
                np.full_like(x, float(case["right_velocity"])),
                np.full_like(x, float(case["right_pressure"])),
            )
        )
        return np.where(
            (x < float(case["discontinuity"]))[:, None],
            np.asarray(case["left"], dtype=np.float64),
            right,
        )
    if kind == "woodward_colella":
        pressure = np.where(
            x < float(case["left_interface"]),
            float(case["left_pressure"]),
            np.where(
                x < float(case["right_interface"]),
                float(case["middle_pressure"]),
                float(case["right_pressure"]),
            ),
        )
        return np.column_stack(
            (
                np.full_like(x, float(case["density"])),
                np.full_like(x, float(case["velocity"])),
                pressure,
            )
        )
    if kind == "smooth_entropy":
        density = float(case["density_base"]) + float(case["density_amplitude"]) * np.sin(
            float(case["wavenumber"]) * x
        )
        return np.column_stack(
            (
                density,
                np.full_like(x, float(case["velocity"])),
                np.full_like(x, float(case["pressure"])),
            )
        )
    raise ValueError(f"unknown case kind: {kind}")


def analytic_reference(case: dict, x: np.ndarray, physical_time: float) -> np.ndarray | None:
    if case["kind"] == "riemann":
        exact = ExactSodSolver(
            left=PrimitiveState(*case["left"]),
            right=PrimitiveState(*case["right"]),
            discontinuity=float(case["discontinuity"]),
        )
        return np.column_stack(exact.sample(x, physical_time))
    if case["kind"] == "smooth_entropy":
        shifted = np.mod(x - float(case["velocity"]) * physical_time, 1.0)
        density = float(case["density_base"]) + float(case["density_amplitude"]) * np.sin(
            float(case["wavenumber"]) * shifted
        )
        return np.column_stack(
            (
                density,
                np.full_like(x, float(case["velocity"])),
                np.full_like(x, float(case["pressure"])),
            )
        )
    return None


def load_fno(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FNO1d(**checkpoint["architecture"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    normalization = {
        key: value.to(device) for key, value in checkpoint["normalization"].items()
    }
    return model, normalization, int(checkpoint["training"]["seed"])


def load_hybrid(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConservativeEulerJumpBlendDKAN(**checkpoint["architecture"]).to(
        device=device, dtype=torch.float64
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return (
        model,
        int(checkpoint["training"]["seed"]),
        str(checkpoint["training"].get("case_family", "unknown")),
    )


def fno_profile(
    model: FNO1d,
    normalization: dict,
    initial: np.ndarray,
    x: np.ndarray,
    physical_time: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    values = np.column_stack(
        (
            np.log(initial[:, 0]),
            initial[:, 1],
            np.log(initial[:, 2]),
            2.0 * x - 1.0,
            np.full_like(x, physical_time),
        )
    ).astype(np.float32)
    tensor = torch.tensor(values[None], device=device)
    normalized = (tensor - normalization["input_mean"]) / normalization["input_std"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        transformed = (
            model(normalized) * normalization["target_std"] + normalization["target_mean"]
        )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    # The clamp is a floating-point guard, not a physical post-processing step.
    # A separate metric records whether the learned logarithmic output reached it.
    density = torch.exp(torch.clamp(transformed[:, 0], -50.0, 50.0))
    pressure = torch.exp(torch.clamp(transformed[:, 2], -50.0, 50.0))
    profile = torch.stack((density, transformed[:, 1], pressure), dim=-1)
    clip_fraction = torch.mean(
        ((torch.abs(transformed[:, 0]) >= 50.0) | (torch.abs(transformed[:, 2]) >= 50.0)).float()
    )
    return profile.detach().cpu().numpy(), seconds, float(clip_fraction.cpu())


def solve_snapshots(
    case: dict,
    cells: int,
    times: np.ndarray,
    device: torch.device,
    cache_root: Path | None,
    default_cfl: float,
) -> tuple[np.ndarray, float]:
    x = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    primitive = initial_primitive(case, x)
    initial = primitive_to_conservative(
        torch.tensor(primitive, dtype=torch.float64, device=device)
    )
    digest = hashlib.sha256(
        initial.detach().cpu().numpy().tobytes()
        + np.asarray(times, dtype=np.float64).tobytes()
        + json.dumps(
            {
                "cells": cells,
                "boundary": case["boundary"],
                "cfl": float(case.get("cfl", default_cfl)),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = cache_root / f"{case['id']}_{digest}_n{cells}.npz" if cache_root else None
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["states"], float(cached["seconds"])

    solver = EulerHLLCFV(
        n_cells=cells,
        boundary=str(case["boundary"]),
        reconstruction="muscl_mc",
        cfl=float(case.get("cfl", default_cfl)),
    )
    state = initial
    states = [state.detach().cpu().numpy()]
    current = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for target in times[1:]:
            while current < float(target) - 1.0e-15:
                dt = min(solver.stable_dt(state), float(target) - current)
                state = solver.step(state, dt)
                current += dt
            states.append(state.detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    stacked = np.stack(states)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, states=stacked, seconds=np.asarray(seconds))
    return stacked, seconds


def interpolate_primitive(states: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    source_x = (np.arange(states.shape[1], dtype=np.float64) + 0.5) / states.shape[1]
    primitive = conservative_to_primitive(torch.tensor(states, dtype=torch.float64)).numpy()
    return np.stack(
        [
            np.column_stack(
                [np.interp(target_x, source_x, snapshot[:, component]) for component in range(3)]
            )
            for snapshot in primitive
        ]
    )


def normalized_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - reference)
        / max(np.linalg.norm(reference), 1.0e-15)
    )


def relative_tv_excess(prediction: np.ndarray, reference: np.ndarray) -> float:
    predicted_tv = float(np.sum(np.abs(np.diff(prediction))))
    reference_tv = float(np.sum(np.abs(np.diff(reference))))
    return max(0.0, predicted_tv - reference_tv) / max(reference_tv, 1.0e-15)


def gradient_concentration_ratio(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    reference_gradient = np.abs(np.diff(reference))
    prediction_gradient = np.abs(np.diff(prediction))
    if float(np.max(reference_gradient)) < 1.0e-10:
        return None
    candidates = np.argsort(reference_gradient)[::-1]
    selected: list[int] = []
    for index in candidates:
        if reference_gradient[index] < 0.1 * float(np.max(reference_gradient)):
            break
        if all(abs(int(index) - prior) >= 16 for prior in selected):
            selected.append(int(index))
        if len(selected) == 4:
            break
    ratios = []
    coordinate = np.arange(reference_gradient.size, dtype=np.float64) + 0.5
    for peak in selected:
        lower = max(0, peak - 24)
        upper = min(reference_gradient.size, peak + 25)
        local_x = coordinate[lower:upper]
        widths = []
        for gradient in (prediction_gradient[lower:upper], reference_gradient[lower:upper]):
            total = float(np.sum(gradient))
            if total <= 1.0e-15:
                widths.append(np.nan)
                continue
            centre = float(np.sum(local_x * gradient) / total)
            widths.append(math.sqrt(float(np.sum((local_x - centre) ** 2 * gradient) / total)))
        if np.isfinite(widths).all() and widths[1] > 1.0e-12:
            ratios.append(widths[0] / widths[1])
    return float(np.mean(ratios)) if ratios else None


def evaluate(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    online_seconds: float,
    *,
    model_seed: int | None = None,
    maximum_cell_average_change: float | None = None,
    fno_log_clip_fraction: float | None = None,
    tv_trust_multiplier: float | None = None,
) -> dict:
    component = [normalized_l2(prediction[:, i], reference[:, i]) for i in range(3)]
    predicted_conservative = primitive_to_conservative(
        torch.tensor(prediction, dtype=torch.float64)
    ).numpy()
    reference_conservative = primitive_to_conservative(
        torch.tensor(reference, dtype=torch.float64)
    ).numpy()
    conservation = np.abs(
        np.mean(predicted_conservative, axis=0) - np.mean(reference_conservative, axis=0)
    ) / np.maximum(np.mean(np.abs(reference_conservative), axis=0), 1.0e-15)
    return {
        "method": method,
        "model_seed": model_seed,
        "density_normalized_l2": component[0],
        "velocity_normalized_l2": component[1],
        "pressure_normalized_l2": component[2],
        "primitive_rms_l2": float(np.sqrt(np.mean(np.square(component)))),
        "primitive_geometric_mean_l2": float(
            np.exp(np.mean(np.log(np.maximum(component, 1.0e-15))))
        ),
        "maximum_relative_conservation_error": float(np.max(conservation)),
        "minimum_density": float(np.min(prediction[:, 0])),
        "minimum_pressure": float(np.min(prediction[:, 2])),
        "density_tv_excess": relative_tv_excess(prediction[:, 0], reference[:, 0]),
        "density_gradient_normalized_l1": float(
            np.sum(np.abs(np.diff(prediction[:, 0]) - np.diff(reference[:, 0])))
            / max(np.sum(np.abs(np.diff(reference[:, 0]))), 1.0e-15)
        ),
        "gradient_concentration_width_ratio": gradient_concentration_ratio(
            prediction[:, 0], reference[:, 0]
        ),
        "online_seconds": online_seconds,
        "maximum_cell_average_change": maximum_cell_average_change,
        "fno_log_clip_fraction": fno_log_clip_fraction,
        "tv_trust_multiplier": tv_trust_multiplier,
        "finite_output": bool(np.isfinite(prediction).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fno-checkpoints", nargs="+", required=True)
    parser.add_argument("--narrow-checkpoints", nargs="+", required=True)
    parser.add_argument("--cse-checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    configuration = protocol["benchmark"]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this Python environment has no CUDA-enabled torch")

    fno_models = [load_fno(Path(path), device) for path in args.fno_checkpoints]
    narrow_models = [load_hybrid(Path(path), device) for path in args.narrow_checkpoints]
    cse_models = [load_hybrid(Path(path), device) for path in args.cse_checkpoints]
    fixed = ConservativeEulerDKANSubcell(hidden_width=16, correction_scale=0.0).to(
        device=device, dtype=torch.float64
    )
    fixed.eval()

    output_dir = Path(args.output_dir)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.state_cache) if args.state_cache else None
    points = int(configuration["evaluation_points"])
    coarse_cells = int(configuration["coarse_cells"])
    fine_cells = int(configuration["fine_comparator_cells"])
    reference_cells = int(configuration["fine_reference_cells"])
    refinement = points // coarse_cells
    if refinement * coarse_cells != points:
        raise ValueError("evaluation points must be divisible by coarse cells")
    coordinates = fixed.uniform_coordinates(refinement, dtype=torch.float64, device=device)
    x = (np.arange(points, dtype=np.float64) + 0.5) / points
    selected_cases = set(args.cases or [])
    cases = [case for case in protocol["cases"] if not selected_cases or case["id"] in selected_cases]
    sensor_enabled = bool(configuration.get("use_discontinuity_sensor", False))
    proposed_method = (
        "sensor_gated_dkan_fv_engineering_tv_trust_0.02"
        if sensor_enabled
        else "physics_gated_dkan_fv_engineering_tv_trust_0.02"
    )
    methods = tuple(
        proposed_method if method == "physics_gated_dkan_fv_engineering_tv_trust_0.02" else method
        for method in BASE_METHODS
    )
    rows: list[dict] = []

    for case in cases:
        final_time = float(case["time"])
        times = np.linspace(0.0, final_time, int(configuration["trajectory_snapshots"]))
        initial_eval = initial_primitive(case, x)
        coarse_states, coarse_seconds = solve_snapshots(
            case, coarse_cells, times, device, cache_root, float(configuration["cfl"])
        )
        fine_states, fine_seconds = solve_snapshots(
            case, fine_cells, times, device, cache_root, float(configuration["cfl"])
        )
        fine_profiles = interpolate_primitive(fine_states, x)
        analytic = analytic_reference(case, x, 0.0)
        if analytic is None:
            case_reference_cells = int(case.get("reference_cells", reference_cells))
            reference_states, reference_seconds = solve_snapshots(
                case, case_reference_cells, times, device, cache_root, float(configuration["cfl"])
            )
            references = interpolate_primitive(reference_states, x)
        else:
            references = np.stack([analytic_reference(case, x, value) for value in times])
            reference_seconds = 0.0

        method_profiles: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        blend_history = []
        trust_history = []
        for time_index, physical_time in enumerate(times):
            coarse_state = torch.tensor(coarse_states[time_index], dtype=torch.float64, device=device)
            with torch.no_grad():
                fixed_conservative = fixed(coarse_state, coordinates)
            fixed_profile = conservative_to_primitive(fixed_conservative).detach().cpu().numpy().reshape(-1, 3)
            piecewise = conservative_to_primitive(
                coarse_state.unsqueeze(-2).expand(-1, refinement, -1)
            ).detach().cpu().numpy().reshape(-1, 3)
            deterministic = {
                "hllc_piecewise_constant": piecewise,
                "hllc_fixed_mc_subcell": fixed_profile,
                "hllc_muscl_fine": fine_profiles[time_index],
            }
            for method, profile in deterministic.items():
                method_profiles[method].append(profile)
                rows.append(
                    {
                        "case_id": case["id"],
                        "case_group": case["group"],
                        "time": float(physical_time),
                        "time_fraction": float(physical_time / final_time) if final_time else 0.0,
                        **evaluate(
                            method,
                            profile,
                            references[time_index],
                            coarse_seconds if method != "hllc_muscl_fine" else fine_seconds,
                        ),
                    }
                )

            narrow_at_time = []
            for model, seed, family in narrow_models:
                started = time.perf_counter()
                with torch.no_grad():
                    reconstructed = model(
                        coarse_state,
                        coordinates,
                        learned_blend_multiplier=float(configuration["hybrid_blend_multiplier"]),
                    )
                elapsed = time.perf_counter() - started
                profile = conservative_to_primitive(reconstructed).detach().cpu().numpy().reshape(-1, 3)
                narrow_at_time.append(profile)
                change = float(torch.max(torch.abs(reconstructed.mean(dim=-2) - coarse_state)).cpu())
                rows.append(
                    {
                        "case_id": case["id"],
                        "case_group": case["group"],
                        "time": float(physical_time),
                        "time_fraction": float(physical_time / final_time) if final_time else 0.0,
                        **evaluate(
                            f"physics_gated_dkan_fv_{family}",
                            profile,
                            references[time_index],
                            coarse_seconds + elapsed,
                            model_seed=seed,
                            maximum_cell_average_change=change,
                        ),
                    }
                )
            method_profiles["physics_gated_dkan_fv_sod_neighbourhood"].append(
                np.median(np.stack(narrow_at_time), axis=0)
            )

            cse_at_time = []
            cse_blends = []
            cse_trust = []
            for model, seed, family in cse_models:
                started = time.perf_counter()
                with torch.no_grad():
                    candidate, blend = model(
                        coarse_state,
                        coordinates,
                        return_blend=True,
                        learned_blend_multiplier=float(configuration["hybrid_blend_multiplier"]),
                        physics_sensor=sensor_enabled,
                    )
                    trusted, multiplier = density_tv_trust_projection(
                        fixed_conservative,
                        candidate,
                        relative_budget=float(configuration["tv_trust_relative_budget"]),
                    )
                elapsed = time.perf_counter() - started
                trusted = trusted.squeeze(0)
                profile = conservative_to_primitive(trusted).detach().cpu().numpy().reshape(-1, 3)
                cse_at_time.append(profile)
                cse_blends.append(blend.detach().cpu().numpy())
                cse_trust.append(float(multiplier.detach().cpu()))
                change = float(torch.max(torch.abs(trusted.mean(dim=-2) - coarse_state)).cpu())
                rows.append(
                    {
                        "case_id": case["id"],
                        "case_group": case["group"],
                        "time": float(physical_time),
                        "time_fraction": float(physical_time / final_time) if final_time else 0.0,
                        **evaluate(
                            (
                                f"sensor_gated_dkan_fv_{family}_tv_trust_0.02"
                                if sensor_enabled
                                else f"physics_gated_dkan_fv_{family}_tv_trust_0.02"
                            ),
                            profile,
                            references[time_index],
                            coarse_seconds + elapsed,
                            model_seed=seed,
                            maximum_cell_average_change=change,
                            tv_trust_multiplier=float(multiplier.detach().cpu()),
                        ),
                    }
                )
            method_profiles[proposed_method].append(
                np.median(np.stack(cse_at_time), axis=0)
            )
            blend_history.append(np.median(np.stack(cse_blends), axis=0))
            trust_history.append(float(np.median(cse_trust)))

            fno_at_time = []
            for model, normalization, seed in fno_models:
                profile, elapsed, clip_fraction = fno_profile(
                    model, normalization, initial_eval, x, float(physical_time), device
                )
                fno_at_time.append(profile)
                rows.append(
                    {
                        "case_id": case["id"],
                        "case_group": case["group"],
                        "time": float(physical_time),
                        "time_fraction": float(physical_time / final_time) if final_time else 0.0,
                        **evaluate(
                            "fno_operator",
                            profile,
                            references[time_index],
                            elapsed,
                            model_seed=seed,
                            fno_log_clip_fraction=clip_fraction,
                        ),
                    }
                )
            method_profiles["fno_operator"].append(np.median(np.stack(fno_at_time), axis=0))

        np.savez_compressed(
            trajectory_dir / f"{case['id']}.npz",
            x=x,
            times=times,
            reference=references,
            initial=initial_eval,
            cse_blend=np.stack(blend_history),
            cse_tv_trust_multiplier=np.asarray(trust_history),
            **{method: np.stack(method_profiles[method]) for method in methods},
        )
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "coarse_seconds": coarse_seconds,
                    "fine_seconds": fine_seconds,
                    "reference_seconds": reference_seconds,
                }
            ),
            flush=True,
        )

    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": torch.__version__,
        "methods": list(methods),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
