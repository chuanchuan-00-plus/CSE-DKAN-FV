"""A-posteriori pretraining of bounded DKAN corrections to WENO5-Z weights.

Training uses random periodic Fourier initial conditions and excludes the
reported sine-wave benchmark.  High-resolution WENO trajectories are used as
the teacher; the recurrent coarse-grid finite-volume rollout is differentiated
end to end.  The pretraining cost is recorded explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cse_dkan.finite_volume import ConservativeBurgersFV, DKANWENO5Reconstruction
from cse_dkan.reference import BurgersWENO


@dataclass(frozen=True)
class TrainingCase:
    viscosity: float
    mean: float
    amplitude_1: float
    amplitude_2: float
    phase_1: float
    phase_2: float

    def values(self, x: np.ndarray) -> np.ndarray:
        return (
            self.mean
            - self.amplitude_1 * np.sin(np.pi * (x - self.phase_1))
            + self.amplitude_2 * np.sin(2.0 * np.pi * (x - self.phase_2))
        )


def make_cases(seed: int, viscosities: list[float], cases_per_viscosity: int) -> list[TrainingCase]:
    rng = np.random.default_rng(seed)
    cases = []
    for viscosity in viscosities:
        for _ in range(cases_per_viscosity):
            cases.append(
                TrainingCase(
                    viscosity=viscosity,
                    mean=float(rng.uniform(-0.25, 0.35)),
                    amplitude_1=float(rng.uniform(0.55, 1.35)),
                    amplitude_2=float(rng.uniform(-0.20, 0.20)),
                    phase_1=float(rng.uniform(-0.75, 0.75)),
                    phase_2=float(rng.uniform(-0.75, 0.75)),
                )
            )
    return cases


def grouped_teacher_data(
    cases: list[TrainingCase], coarse_cells: int, refinement: int, final_time: float
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    fine_cells = coarse_cells * refinement
    grouped: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}
    for case in cases:
        solver = BurgersWENO(
            x_left=-1.0,
            x_right=1.0,
            n_cells=fine_cells,
            viscosity=case.viscosity,
        )
        fine_initial = case.values(solver.x)
        fine_target = solver.solve(case.values, final_time)[final_time]
        coarse_initial = fine_initial.reshape(coarse_cells, refinement).mean(axis=1)
        coarse_target = fine_target.reshape(coarse_cells, refinement).mean(axis=1)
        grouped.setdefault(case.viscosity, []).append((coarse_initial, coarse_target))
    return {
        viscosity: (
            torch.tensor(np.stack([row[0] for row in rows]), dtype=torch.float32),
            torch.tensor(np.stack([row[1] for row in rows]), dtype=torch.float32),
        )
        for viscosity, rows in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--coarse-cells", type=int, default=64)
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--cases-per-viscosity", type=int, default=4)
    parser.add_argument("--final-time", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The held-out benchmark viscosity 1e-3 is deliberately omitted.
    viscosities = [0.0, 3.0e-4, 3.0e-3]
    cases = make_cases(args.seed, viscosities, args.cases_per_viscosity)
    generation_started = time.perf_counter()
    data = grouped_teacher_data(cases, args.coarse_cells, args.refinement, args.final_time)
    generation_seconds = time.perf_counter() - generation_started
    data = {key: (value[0].to(device), value[1].to(device)) for key, value in data.items()}

    model = DKANWENO5Reconstruction(hidden_width=8, correction_scale=0.35).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        viscosity = viscosities[(iteration - 1) % len(viscosities)]
        initial, target = data[viscosity]
        solver = ConservativeBurgersFV(
            n_cells=args.coarse_cells,
            cfl=0.20,
            limiter=model,
            viscosity=viscosity,
        )
        prediction = solver.solve(initial, args.final_time)
        relative_mse = torch.mean((prediction - target) ** 2) / (
            torch.mean(target**2) + 1.0e-6
        )
        target_min = torch.amin(target, dim=-1, keepdim=True)
        target_max = torch.amax(target, dim=-1, keepdim=True)
        bound_penalty = torch.mean(
            torch.relu(target_min - prediction) ** 2
            + torch.relu(prediction - target_max) ** 2
        )
        loss = relative_mse + 10.0 * bound_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            row = {
                "iteration": float(iteration),
                "viscosity": viscosity,
                "relative_mse": float(relative_mse.detach()),
                "bound_penalty": float(bound_penalty.detach()),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": {
                "hidden_width": 8,
                "correction_scale": 0.35,
            },
            "training": vars(args),
            "training_cases": [asdict(case) for case in cases],
            "history": history,
            "generation_seconds": generation_seconds,
            "training_seconds": training_seconds,
            "device": str(device),
        },
        output,
    )
    metadata = output.with_suffix(".json")
    with metadata.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "training": vars(args),
                "training_cases": [asdict(case) for case in cases],
                "history": history,
                "generation_seconds": generation_seconds,
                "training_seconds": training_seconds,
                "device": str(device),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
