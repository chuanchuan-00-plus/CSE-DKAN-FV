"""Train DKAN to blend a conservative MC profile with an explicit Euler jump basis."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.euler_finite_volume import conservative_to_primitive
from cse_dkan.euler_subcell import ConservativeEulerJumpBlendDKAN
from train_euler_subcell_dkan import (
    RiemannCase,
    generate_records,
    make_cases,
    normalized_primitive_mse,
)


def make_sod_neighbourhood_cases(seed: int, count: int) -> list[RiemannCase]:
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        cases.append(
            RiemannCase(
                left_density=float(rng.uniform(0.78, 1.22)),
                left_velocity=float(rng.uniform(-0.08, 0.08)),
                left_pressure=float(rng.uniform(0.78, 1.22)),
                right_density=float(rng.uniform(0.09, 0.20)),
                right_velocity=float(rng.uniform(-0.08, 0.08)),
                right_pressure=float(rng.uniform(0.065, 0.16)),
                validation=index >= count - max(2, count // 4),
            )
        )
    return cases


def make_engineering_cases(seed: int, count: int) -> list[RiemannCase]:
    """Match the broad parametric range used by the FNO operator baseline."""
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        cases.append(
            RiemannCase(
                left_density=float(rng.uniform(0.50, 2.00)),
                left_velocity=float(rng.uniform(-0.35, 0.35)),
                left_pressure=float(rng.uniform(0.35, 2.00)),
                right_density=float(rng.uniform(0.05, 0.60)),
                right_velocity=float(rng.uniform(-0.35, 0.35)),
                right_pressure=float(rng.uniform(0.02, 0.32)),
                validation=index >= count - max(2, count // 4),
            )
        )
    return cases


def records_to_device(records: list[dict], device: torch.device, dtype: torch.dtype) -> list[dict]:
    converted = []
    for record in records:
        converted.append(
            {
                key: value.to(device=device, dtype=dtype)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in record.items()
            }
        )
    return converted


@torch.no_grad()
def oracle_dataset(model, records, coordinates, levels: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    all_features = []
    all_targets = []
    alpha = torch.linspace(
        0.0,
        1.0,
        levels,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    for record in records:
        coarse = record["coarse"]
        fixed, jump = model.basis_profiles(coarse, coordinates)
        expanded_coarse = coarse.unsqueeze(0).expand(levels, -1, -1)
        expanded_fixed = fixed.unsqueeze(0).expand(levels, -1, -1, -1)
        expanded_jump = jump.unsqueeze(0).expand(levels, -1, -1, -1)
        blends = alpha.unsqueeze(-1).expand(levels, coarse.shape[-2])
        candidates = model.reconstruct_from_blend(
            expanded_coarse,
            coordinates,
            blends,
            basis=(expanded_fixed, expanded_jump),
        )
        primitive = conservative_to_primitive(candidates)
        target = record["target"].unsqueeze(0)
        scale = torch.sqrt(torch.mean(target**2, dim=(-3, -2), keepdim=True)).clamp_min(
            1.0e-3
        )
        error = torch.mean(((primitive - target) / scale) ** 2, dim=(-2, -1))
        oracle_index = torch.argmin(error, dim=0)
        all_features.append(model.blend_features(coarse))
        all_targets.append(alpha[oracle_index])
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def profile_validation_loss(model, records, coordinates) -> float:
    errors = []
    for record in records:
        primitive = conservative_to_primitive(model(record["coarse"], coordinates))
        errors.append(normalized_primitive_mse(primitive, record["target"]))
    return float(torch.mean(torch.stack(errors)).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--coarse-cells", nargs="+", type=int, default=[64, 96, 128])
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=16180)
    parser.add_argument("--dataset-seed", type=int)
    parser.add_argument("--record-cache")
    parser.add_argument(
        "--case-family",
        choices=("broad", "engineering", "sod_neighbourhood"),
        default="broad",
    )
    parser.add_argument("--basis-type", choices=("shock", "contact"), default="shock")
    parser.add_argument("--times", nargs="+", type=float, default=[0.10, 0.20])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    dataset_seed = args.seed if args.dataset_seed is None else args.dataset_seed
    if args.case_family == "broad":
        cases = make_cases(dataset_seed, args.cases)
    elif args.case_family == "engineering":
        cases = make_engineering_cases(dataset_seed, args.cases)
    else:
        cases = make_sod_neighbourhood_cases(dataset_seed, args.cases)
    generation_started = time.perf_counter()
    data_configuration = {
        "case_family": args.case_family,
        "dataset_seed": dataset_seed,
        "cases": args.cases,
        "coarse_cells": args.coarse_cells,
        "refinement": args.refinement,
        "times": args.times,
        "dtype": args.dtype,
    }
    cache_path = Path(args.record_cache) if args.record_cache else None
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["configuration"] != data_configuration:
            raise ValueError("record cache configuration does not match this run")
        records = records_to_device(cached["records"], device, dtype)
    else:
        records = []
        for cells in args.coarse_cells:
            records.extend(
                generate_records(
                    cases,
                    cells,
                    args.refinement,
                    tuple(args.times),
                    device,
                    dtype,
                )
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "configuration": data_configuration,
                    "records": records_to_device(records, torch.device("cpu"), dtype),
                },
                cache_path,
            )
    generation_seconds = time.perf_counter() - generation_started
    training_records = [record for record in records if not record["validation"]]
    validation_records = [record for record in records if record["validation"]]

    model = ConservativeEulerJumpBlendDKAN(
        hidden_width=16, basis_type=args.basis_type
    ).to(device=device, dtype=dtype)
    coordinates = model.uniform_coordinates(
        args.refinement, dtype=dtype, device=device
    )
    oracle_started = time.perf_counter()
    training_features, training_targets = oracle_dataset(
        model, training_records, coordinates
    )
    validation_features, validation_targets = oracle_dataset(
        model, validation_records, coordinates
    )
    oracle_seconds = time.perf_counter() - oracle_started
    fixed_profile_validation_mse = profile_validation_loss(
        model, validation_records, coordinates
    )

    optimizer = torch.optim.Adam(model.blend_network.parameters(), lr=1.0e-3)
    rng = np.random.default_rng(args.seed + 1)
    history = []
    best_profile_validation_mse = fixed_profile_validation_mse
    best_iteration = 0
    best_state = copy.deepcopy(model.state_dict())
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        indices = torch.tensor(
            rng.integers(training_targets.numel(), size=args.batch_size),
            dtype=torch.long,
            device=device,
        )
        prediction = torch.sigmoid(
            model.blend_network(training_features[indices]).reshape(-1)
        )
        target = training_targets[indices]
        weight = 1.0 + 5.0 * (target > 0.025).to(target.dtype)
        loss = torch.mean(weight * (prediction - target) ** 2) / torch.mean(weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.blend_network.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
            with torch.no_grad():
                validation_prediction = torch.sigmoid(
                    model.blend_network(validation_features).reshape(-1)
                )
                validation_oracle_mse = float(
                    torch.mean((validation_prediction - validation_targets) ** 2).cpu()
                )
                profile_mse = profile_validation_loss(
                    model, validation_records, coordinates
                )
            row = {
                "iteration": iteration,
                "training_oracle_mse": float(loss.detach().cpu()),
                "validation_oracle_mse": validation_oracle_mse,
                "validation_profile_mse": profile_mse,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if profile_mse < best_profile_validation_mse:
                best_profile_validation_mse = profile_mse
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)

    payload = {
        "model_class": "ConservativeEulerJumpBlendDKAN",
        "model_state": model.state_dict(),
        "architecture": {
            "hidden_width": 16,
            "gamma": 1.4,
            "positivity_floor": 1.0e-8,
            "basis_type": args.basis_type,
        },
        "training": vars(args),
        "data_configuration": data_configuration,
        "cases": [case.__dict__ for case in cases],
        "history": history,
        "fixed_profile_validation_mse": fixed_profile_validation_mse,
        "best_profile_validation_mse": best_profile_validation_mse,
        "best_iteration": best_iteration,
        "final_profile_validation_mse": profile_validation_loss(
            model, validation_records, coordinates
        ),
        "generation_seconds": generation_seconds,
        "oracle_seconds": oracle_seconds,
        "training_seconds": training_seconds,
        "device": str(device),
        "dtype": args.dtype,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in payload.items() if key != "model_state"},
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
