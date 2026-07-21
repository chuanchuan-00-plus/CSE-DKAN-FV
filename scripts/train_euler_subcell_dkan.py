"""Train positivity-preserving conservative DKAN subcells on generic Riemann flows."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cse_dkan.euler_finite_volume import (
    EulerHLLCFV,
    conservative_to_primitive,
    primitive_to_conservative,
)
from cse_dkan.euler_subcell import ConservativeEulerDKANSubcell
from cse_dkan.reference import ExactSodSolver, PrimitiveState


@dataclass(frozen=True)
class RiemannCase:
    left_density: float
    left_velocity: float
    left_pressure: float
    right_density: float
    right_velocity: float
    right_pressure: float
    validation: bool

    @property
    def left(self) -> PrimitiveState:
        return PrimitiveState(self.left_density, self.left_velocity, self.left_pressure)

    @property
    def right(self) -> PrimitiveState:
        return PrimitiveState(self.right_density, self.right_velocity, self.right_pressure)


def make_cases(seed: int, count: int) -> list[RiemannCase]:
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        cases.append(
            RiemannCase(
                left_density=float(rng.uniform(0.65, 1.35)),
                left_velocity=float(rng.uniform(-0.18, 0.18)),
                left_pressure=float(rng.uniform(0.65, 1.45)),
                right_density=float(rng.uniform(0.10, 0.65)),
                right_velocity=float(rng.uniform(-0.18, 0.18)),
                right_pressure=float(rng.uniform(0.07, 0.55)),
                validation=index >= count - max(2, count // 4),
            )
        )
    return cases


def initial_state(
    case: RiemannCase,
    cells: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = (torch.arange(cells, dtype=dtype, device=device) + 0.5) / cells
    left = torch.tensor(
        [case.left_density, case.left_velocity, case.left_pressure],
        dtype=dtype,
        device=device,
    )
    right = torch.tensor(
        [case.right_density, case.right_velocity, case.right_pressure],
        dtype=dtype,
        device=device,
    )
    primitive = torch.where((x < 0.5).unsqueeze(-1), left, right)
    return primitive_to_conservative(primitive)


def generate_records(
    cases: list[RiemannCase],
    cells: int,
    refinement: int,
    times: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> list[dict]:
    offsets = (np.arange(refinement) + 0.5) / refinement
    fine_x = ((np.arange(cells)[:, None] + offsets[None, :]) / cells).reshape(-1)
    records = []
    for case in cases:
        exact = ExactSodSolver(left=case.left, right=case.right)
        initial = initial_state(case, cells, device, dtype)
        solver = EulerHLLCFV(
            n_cells=cells,
            boundary="outflow",
            reconstruction="muscl_mc",
            cfl=0.25,
        )
        for final_time in times:
            with torch.no_grad():
                coarse = solver.solve(initial, final_time)
            density, velocity, pressure = exact.sample(fine_x, final_time)
            target = torch.tensor(
                np.column_stack((density, velocity, pressure)).reshape(
                    cells, refinement, 3
                ),
                dtype=dtype,
                device=device,
            )
            records.append(
                {
                    "coarse": coarse,
                    "target": target,
                    "time": final_time,
                    "validation": case.validation,
                }
            )
    return records


def normalized_primitive_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(torch.mean(target**2, dim=(-3, -2), keepdim=True)).clamp_min(1.0e-3)
    return torch.mean(((prediction - target) / scale) ** 2)


@torch.no_grad()
def dataset_loss(model, records, coordinates) -> float:
    errors = []
    for record in records:
        conservative = model(record["coarse"], coordinates)
        primitive = conservative_to_primitive(conservative)
        errors.append(normalized_primitive_mse(primitive, record["target"]))
    return float(torch.mean(torch.stack(errors)).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--coarse-cells", type=int, default=80)
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=27182)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    cases = make_cases(args.seed, args.cases)
    generation_started = time.perf_counter()
    records = generate_records(
        cases,
        args.coarse_cells,
        args.refinement,
        (0.08, 0.14, 0.20),
        device,
        dtype,
    )
    generation_seconds = time.perf_counter() - generation_started
    training_records = [record for record in records if not record["validation"]]
    validation_records = [record for record in records if record["validation"]]

    model = ConservativeEulerDKANSubcell(
        hidden_width=16, correction_scale=0.5
    ).to(device=device, dtype=dtype)
    fixed = ConservativeEulerDKANSubcell(
        hidden_width=16, correction_scale=0.0
    ).to(device=device, dtype=dtype)
    coordinates = model.uniform_coordinates(
        args.refinement, dtype=dtype, device=device
    )
    fixed_validation_mse = dataset_loss(fixed, validation_records, coordinates)
    with torch.no_grad():
        for record in records:
            record["fixed_primitive"] = conservative_to_primitive(
                fixed(record["coarse"], coordinates)
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    rng = np.random.default_rng(args.seed + 1)
    initial_validation_mse = dataset_loss(model, validation_records, coordinates)
    history = [
        {
            "iteration": 0,
            "training_weighted_mse": None,
            "validation_mse": initial_validation_mse,
        }
    ]
    best_validation_mse = initial_validation_mse
    best_iteration = 0
    best_state = copy.deepcopy(model.state_dict())
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        batch_losses = []
        batch_indices = rng.integers(len(training_records), size=args.batch_size)
        for record_index in batch_indices:
            record = training_records[int(record_index)]
            reconstructed = model(record["coarse"], coordinates)
            primitive = conservative_to_primitive(reconstructed)
            target = record["target"]
            flat_target = target.reshape(-1, 3)
            gradient = torch.abs(flat_target - torch.roll(flat_target, 1, dims=0))
            weight = 1.0 + 2.0 * torch.sum(gradient, dim=-1, keepdim=True) / (
                torch.mean(torch.sum(gradient, dim=-1)) + 1.0e-5
            )
            scale = torch.sqrt(
                torch.mean(target**2, dim=(-3, -2), keepdim=True)
            ).clamp_min(1.0e-3)
            weighted_error = weight.reshape(*target.shape[:-1], 1) * (
                (primitive - target) / scale
            ) ** 2
            data_loss = torch.mean(weighted_error) / torch.mean(weight)
            residual_loss = torch.mean(
                ((primitive - record["fixed_primitive"]) / scale) ** 2
            )
            batch_losses.append(data_loss + args.residual_weight * residual_loss)
        loss = torch.mean(torch.stack(batch_losses))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % 50 == 0 or iteration == args.iterations:
            validation_mse = dataset_loss(model, validation_records, coordinates)
            row = {
                "iteration": iteration,
                "training_weighted_mse": float(loss.detach().cpu()),
                "validation_mse": validation_mse,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)

    payload = {
        "model_state": model.state_dict(),
        "architecture": {
            "hidden_width": 16,
            "correction_scale": 0.5,
            "gamma": 1.4,
            "positivity_floor": 1.0e-8,
        },
        "training": vars(args),
        "cases": [asdict(case) for case in cases],
        "history": history,
        "fixed_validation_mse": fixed_validation_mse,
        "initial_validation_mse": initial_validation_mse,
        "best_validation_mse": best_validation_mse,
        "best_iteration": best_iteration,
        "final_validation_mse": dataset_loss(model, validation_records, coordinates),
        "generation_seconds": generation_seconds,
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
