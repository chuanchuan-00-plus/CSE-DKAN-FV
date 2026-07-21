"""Train the hard-conservative DKAN subcell reconstruction on generic flows."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cse_dkan.finite_volume import ConservativeBurgersFV
from cse_dkan.reference import BurgersWENO
from cse_dkan.subcell import ConservativeDKANSubcell


@dataclass(frozen=True)
class Case:
    viscosity: float
    mean: float
    amplitude_1: float
    amplitude_2: float
    phase_1: float
    phase_2: float
    validation: bool

    def values(self, x: np.ndarray) -> np.ndarray:
        return (
            self.mean
            - self.amplitude_1 * np.sin(np.pi * (x - self.phase_1))
            + self.amplitude_2 * np.sin(2.0 * np.pi * (x - self.phase_2))
        )


def make_cases(
    seed: int,
    cases_per_viscosity: int,
    amplitude_min: float = 0.55,
    amplitude_max: float = 1.35,
) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []
    # The target viscosity 1e-3 and the target sine initial condition are held out.
    for viscosity in (0.0, 3.0e-4, 3.0e-3, 1.0e-2):
        for index in range(cases_per_viscosity):
            cases.append(
                Case(
                    viscosity=viscosity,
                    mean=float(rng.uniform(-0.25, 0.35)),
                    amplitude_1=float(rng.uniform(amplitude_min, amplitude_max)),
                    amplitude_2=float(rng.uniform(-0.20, 0.20)),
                    phase_1=float(rng.uniform(-0.75, 0.75)),
                    phase_2=float(rng.uniform(-0.75, 0.75)),
                    validation=index
                    >= cases_per_viscosity - max(1, cases_per_viscosity // 4),
                )
            )
    return cases


def generate_records(
    cases: list[Case],
    coarse_cells: int,
    refinement: int,
    times: tuple[float, ...],
    dtype: torch.dtype,
) -> list[dict]:
    records = []
    fine_cells = coarse_cells * refinement
    for case in cases:
        teacher = BurgersWENO(
            x_left=-1.0,
            x_right=1.0,
            n_cells=fine_cells,
            viscosity=case.viscosity,
        )
        fine_initial = case.values(teacher.x)
        teacher_snapshots = teacher.solve(case.values, max(times), times)
        coarse_initial = torch.tensor(
            fine_initial.reshape(coarse_cells, refinement).mean(axis=1), dtype=dtype
        )
        coarse_solver = ConservativeBurgersFV(
            n_cells=coarse_cells, limiter="tvd", viscosity=case.viscosity
        )
        for final_time in times:
            coarse = coarse_solver.solve(coarse_initial, final_time)
            target = torch.tensor(
                teacher_snapshots[final_time].reshape(coarse_cells, refinement),
                dtype=dtype,
            )
            records.append(
                {
                    "coarse": coarse,
                    "target": target,
                    "viscosity": case.viscosity,
                    "time": final_time,
                    "validation": case.validation,
                }
            )
    return records


@torch.no_grad()
def dataset_loss(model, records, coordinates, dx) -> float:
    errors = []
    for record in records:
        prediction = model(
            record["coarse"], coordinates, viscosity=record["viscosity"], dx=dx
        )
        errors.append(torch.mean((prediction - record["target"]) ** 2))
    return float(torch.mean(torch.stack(errors)).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--coarse-cells", type=int, default=64)
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--cases-per-viscosity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--tv-weight", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--amplitude-min", type=float, default=0.55)
    parser.add_argument("--amplitude-max", type=float, default=1.35)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    if not 0.0 < args.amplitude_min < args.amplitude_max:
        raise ValueError("amplitude range must satisfy 0 < minimum < maximum")
    cases = make_cases(
        args.seed,
        args.cases_per_viscosity,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max,
    )
    generation_started = time.perf_counter()
    records = generate_records(
        cases,
        args.coarse_cells,
        args.refinement,
        (0.20, 0.35, 0.50),
        dtype,
    )
    generation_seconds = time.perf_counter() - generation_started
    for record in records:
        record["coarse"] = record["coarse"].to(device=device, dtype=dtype)
        record["target"] = record["target"].to(device=device, dtype=dtype)
    training_records = [record for record in records if not record["validation"]]
    validation_records = [record for record in records if record["validation"]]

    model = ConservativeDKANSubcell(hidden_width=16, correction_scale=0.50).to(
        device=device, dtype=dtype
    )
    coordinates = model.uniform_coordinates(
        args.refinement, dtype=dtype, device=device
    )
    fixed = ConservativeDKANSubcell(hidden_width=16, correction_scale=0.0).to(
        device=device, dtype=dtype
    ).eval()
    dx = 2.0 / args.coarse_cells
    fixed_validation_mse = dataset_loss(fixed, validation_records, coordinates, dx)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    rng = np.random.default_rng(args.seed + 1)
    history = []
    best_validation_mse = float("inf")
    best_iteration = 0
    best_state = copy.deepcopy(model.state_dict())
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        record = training_records[int(rng.integers(len(training_records)))]
        prediction = model(
            record["coarse"],
            coordinates,
            viscosity=record["viscosity"],
            dx=dx,
        )
        target = record["target"]
        gradient = torch.abs(target - torch.roll(target, 1, dims=-1))
        weight = 1.0 + 3.0 * gradient / (torch.mean(gradient) + 1.0e-5)
        weighted_mse = torch.mean(weight * (prediction - target) ** 2) / torch.mean(weight)
        prediction_flat = prediction.reshape(-1)
        target_flat = target.reshape(-1)
        prediction_tv = torch.sum(
            torch.abs(prediction_flat - torch.roll(prediction_flat, 1))
        )
        target_tv = torch.sum(torch.abs(target_flat - torch.roll(target_flat, 1)))
        tv_excess_penalty = (
            torch.relu(prediction_tv - target_tv) / (target_tv + 1.0e-5)
        ) ** 2
        loss = weighted_mse + args.tv_weight * tv_excess_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
            validation_mse = dataset_loss(model, validation_records, coordinates, dx)
            row = {
                "iteration": float(iteration),
                "training_weighted_mse": float(weighted_mse.detach()),
                "training_tv_excess_penalty": float(tv_excess_penalty.detach()),
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "architecture": {"hidden_width": 16, "correction_scale": 0.50},
        "training": vars(args),
        "cases": [asdict(case) for case in cases],
        "history": history,
        "fixed_validation_mse": fixed_validation_mse,
        "best_validation_mse": best_validation_mse,
        "best_iteration": best_iteration,
        "final_validation_mse": dataset_loss(
            model, validation_records, coordinates, dx
        ),
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "device": str(device),
        "dtype": args.dtype,
    }
    torch.save(payload, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in payload.items() if key != "model_state"}, handle, indent=2)


if __name__ == "__main__":
    main()
