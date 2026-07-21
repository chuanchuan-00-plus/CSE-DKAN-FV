"""Train a PDEBench-style FNO on a parametric family of exact Euler Riemann solutions."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cse_dkan.fno import FNO1d
from cse_dkan.reference import ExactSodSolver, PrimitiveState


RANGES = {
    "left_density": (0.50, 2.00),
    "left_velocity": (-0.35, 0.35),
    "left_pressure": (0.35, 2.00),
    "right_density": (0.05, 0.60),
    "right_velocity": (-0.35, 0.35),
    "right_pressure": (0.02, 0.32),
    "time": (0.04, 0.32),
}


def sample_parameters(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = tuple(RANGES)
    columns = [rng.uniform(*RANGES[name], size=count) for name in names]
    return np.column_stack(columns)


def make_dataset(parameters: np.ndarray, points: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = (np.arange(points, dtype=np.float64) + 0.5) / points
    inputs = np.empty((len(parameters), points, 5), dtype=np.float32)
    targets = np.empty((len(parameters), points, 3), dtype=np.float32)
    for index, row in enumerate(parameters):
        rho_l, u_l, p_l, rho_r, u_r, p_r, final_time = row
        left = PrimitiveState(rho_l, u_l, p_l)
        right = PrimitiveState(rho_r, u_r, p_r)
        exact = ExactSodSolver(left=left, right=right)
        density, velocity, pressure = exact.sample(x, final_time)
        initial_density = np.where(x < 0.5, rho_l, rho_r)
        initial_velocity = np.where(x < 0.5, u_l, u_r)
        initial_pressure = np.where(x < 0.5, p_l, p_r)
        inputs[index] = np.column_stack(
            (
                np.log(initial_density),
                initial_velocity,
                np.log(initial_pressure),
                2.0 * x - 1.0,
                np.full_like(x, final_time),
            )
        )
        targets[index] = np.column_stack(
            (np.log(density), velocity, np.log(pressure))
        )
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def normalize(
    values: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor
) -> torch.Tensor:
    return (values - mean) / standard_deviation


@torch.no_grad()
def validation_loss(
    model: FNO1d,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    losses = []
    for start in range(0, inputs.shape[0], batch_size):
        batch_input = normalize(
            inputs[start : start + batch_size].to(device), input_mean, input_std
        )
        batch_target = normalize(
            targets[start : start + batch_size].to(device), target_mean, target_std
        )
        losses.append(torch.mean((model(batch_input) - batch_target) ** 2))
    return float(torch.mean(torch.stack(losses)).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-seed", type=int, default=20260716)
    parser.add_argument("--train-cases", type=int, default=512)
    parser.add_argument("--validation-cases", type=int, default=128)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation_started = time.perf_counter()
    train_parameters = sample_parameters(args.train_cases, args.dataset_seed)
    validation_parameters = sample_parameters(
        args.validation_cases, args.dataset_seed + 1
    )
    train_inputs, train_targets = make_dataset(train_parameters, args.points)
    validation_inputs, validation_targets = make_dataset(
        validation_parameters, args.points
    )
    generation_seconds = time.perf_counter() - generation_started
    input_mean = train_inputs.mean(dim=(0, 1), keepdim=True).to(device)
    input_std = train_inputs.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-5).to(device)
    target_mean = train_targets.mean(dim=(0, 1), keepdim=True).to(device)
    target_std = train_targets.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-5).to(device)
    model = FNO1d(
        width=args.width, modes=args.modes, layers=args.layers
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=5.0e-5
    )
    rng = np.random.default_rng(args.seed + 17)
    best_validation = float("inf")
    best_iteration = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        indices = torch.as_tensor(
            rng.integers(0, args.train_cases, size=args.batch_size), dtype=torch.long
        )
        batch_input = normalize(
            train_inputs[indices].to(device), input_mean, input_std
        )
        batch_target_raw = train_targets[indices].to(device)
        batch_target = normalize(
            batch_target_raw, target_mean, target_std
        )
        prediction = model(batch_input)
        density_gradient = torch.abs(
            batch_target_raw[:, 1:, 0] - batch_target_raw[:, :-1, 0]
        )
        density_gradient = torch.nn.functional.pad(density_gradient, (0, 1))
        weight = 1.0 + 3.0 * density_gradient / (
            density_gradient.mean(dim=1, keepdim=True) + 1.0e-5
        )
        loss = torch.mean(weight.unsqueeze(-1) * (prediction - batch_target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
            value = validation_loss(
                model,
                validation_inputs,
                validation_targets,
                input_mean,
                input_std,
                target_mean,
                target_std,
                16,
                device,
            )
            row = {
                "iteration": iteration,
                "training_weighted_mse": float(loss.detach()),
                "validation_mse": value,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if value < best_validation:
                best_validation = value
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    payload = {
        "model_state": model.state_dict(),
        "architecture": {
            "in_channels": 5,
            "out_channels": 3,
            "width": args.width,
            "modes": args.modes,
            "layers": args.layers,
        },
        "normalization": {
            "input_mean": input_mean.cpu(),
            "input_std": input_std.cpu(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
        },
        "training": vars(args),
        "ranges": RANGES,
        "history": history,
        "best_iteration": best_iteration,
        "best_validation_mse": best_validation,
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "device": str(device),
        "dtype": "float32",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in payload.items() if key not in {"model_state", "normalization"}},
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
