"""Fair MLP/DKAN/weak-conservation PINN development loop for Euler Sod."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .euler_finite_volume import euler_flux, primitive_to_conservative
from .euler_losses import (
    euler_control_volume_and_entropy,
    euler_physical_entropy_violation,
    euler_rankine_hugoniot_residual,
    euler_strong_residual,
)
from .losses import RectangularControlVolumes, gradient
from .metrics import normalized_lp, shock_location_from_gradient, shock_width_10_90
from .models import (
    AffineCoordinateMap,
    DKAN,
    EulerWaveExplicitAnsatz,
    MLP,
    PositivePrimitiveWrapper,
    trainable_parameter_count,
)
from .reference import ExactSodSolver


@dataclass
class EulerTrainingConfig:
    variant: str = "mlp_pinn"
    seed: int = 0
    steps: int = 1000
    batch_interior: int = 512
    batch_initial: int = 256
    batch_boundary: int = 128
    batch_control_volumes: int = 64
    quadrature_order: int = 4
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 5.0e-5
    weight_pde: float = 1.0
    weight_initial: float = 50.0
    weight_boundary: float = 10.0
    weight_cv: float = 50.0
    weight_entropy: float = 10.0
    weight_global: float = 200.0
    weight_rh: float = 10.0
    weight_contact: float = 10.0
    weight_shock_entropy: float = 5.0
    weight_separation: float = 10.0
    weight_trace: float = 20.0
    freeze_wave_geometry: bool = True
    dtype: str = "float32"
    device: str = "cuda"
    log_every: int = 100
    evaluation_points: int = 2000
    artificial_viscosity: float = 0.005
    gradient_weight_momentum: float = 0.9
    gradient_weight_minimum: float = 0.1
    gradient_weight_maximum: float = 10.0

    @classmethod
    def from_json(cls, path: str | Path) -> "EulerTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unknown dtype: {name}")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_euler_model(variant: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    coordinate_map = AffineCoordinateMap(0.0, 1.0, 0.2)
    if variant in {"mlp_pinn", "gw_pinn", "av_pinn", "ci_pinn"}:
        network = MLP([2, 64, 64, 64, 3])
    elif variant in {"dkan_pinn", "cw_dkan"}:
        network = DKAN([2, 24, 24, 3], embedding_frequencies=12)
    elif variant == "cse_dkan":
        exact = ExactSodSolver()
        gamma = exact.gamma
        sound_right = math.sqrt(gamma * exact.right.pressure / exact.right.density)
        shock_speed = exact.right.velocity + sound_right * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * exact.p_star / exact.right.pressure
            + (gamma - 1.0) / (2.0 * gamma)
        )
        model = EulerWaveExplicitAnsatz(
            DKAN([2, 22, 22, 9], embedding_frequencies=12),
            coordinate_map,
            contact_speed=exact.u_star,
            shock_speed=shock_speed,
        )
        return model.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown Euler variant: {variant}")
    return PositivePrimitiveWrapper(nn.Sequential(coordinate_map, network)).to(device=device, dtype=dtype)


def _uniform(count: int, low: float, high: float, device, dtype) -> Tensor:
    return low + (high - low) * torch.rand((count, 1), device=device, dtype=dtype)


def _sod_initial(x: Tensor) -> Tensor:
    left = torch.tensor([1.0, 0.0, 1.0], dtype=x.dtype, device=x.device)
    right = torch.tensor([0.125, 0.0, 0.1], dtype=x.dtype, device=x.device)
    return torch.where((x < 0.5).expand(-1, 3), left, right)


def _initial_loss(model, config, device, dtype) -> Tensor:
    random_x = _uniform(config.batch_initial // 2, 0.0, 1.0, device, dtype)
    near = 0.5 + 0.08 * (2.0 * torch.rand((config.batch_initial - random_x.shape[0], 1), device=device, dtype=dtype) - 1.0)
    x = torch.cat((random_x, near), dim=0)
    points = torch.cat((x, torch.zeros_like(x)), dim=-1)
    return F.mse_loss(model(points), _sod_initial(x))


def _boundary_loss(model, config, max_time, device, dtype) -> Tensor:
    times = _uniform(config.batch_boundary, 0.0, max_time, device, dtype)
    left_points = torch.cat((torch.zeros_like(times), times), dim=-1)
    right_points = torch.cat((torch.ones_like(times), times), dim=-1)
    left = torch.tensor([1.0, 0.0, 1.0], dtype=dtype, device=device).expand(times.shape[0], 3)
    right = torch.tensor([0.125, 0.0, 0.1], dtype=dtype, device=device).expand(times.shape[0], 3)
    return F.mse_loss(model(left_points), left) + F.mse_loss(model(right_points), right)


def _volumes(config, max_time, device, dtype) -> RectangularControlVolumes:
    n = config.batch_control_volumes
    hx = _uniform(n, 0.01, 0.05, device, dtype)
    maximum_ht = min(0.012, 0.2 * max_time)
    ht = _uniform(n, 0.002, max(maximum_ht, 0.0021), device, dtype)
    x = hx + (1.0 - 2.0 * hx) * torch.rand_like(hx)
    t = ht + (max_time - 2.0 * ht).clamp_min(1.0e-5) * torch.rand_like(ht)
    return RectangularControlVolumes(torch.cat((x, t), dim=-1), hx, ht)


def _global_balance(model, max_time, device, dtype) -> Tensor:
    x = torch.linspace(0.0, 1.0, 256, device=device, dtype=dtype).reshape(1, -1, 1)
    times = torch.linspace(0.0, max_time, 5, device=device, dtype=dtype)[1:].reshape(-1, 1, 1)
    points = torch.cat((x.expand(times.shape[0], -1, -1), times.expand(-1, x.shape[1], -1)), dim=-1)
    conservative = primitive_to_conservative(model(points.reshape(-1, 2))).reshape(times.shape[0], x.shape[1], 3)
    predicted = conservative.mean(dim=1)
    initial_x = x.reshape(-1, 1)
    initial = primitive_to_conservative(_sod_initial(initial_x)).mean(dim=0, keepdim=True)
    left = primitive_to_conservative(torch.tensor([[1.0, 0.0, 1.0]], dtype=dtype, device=device))
    right = primitive_to_conservative(torch.tensor([[0.125, 0.0, 0.1]], dtype=dtype, device=device))
    boundary_rate = euler_flux(left) - euler_flux(right)
    expected = initial + times.reshape(-1, 1) * boundary_rate
    scale = torch.tensor([1.0, 1.0, 2.5], dtype=dtype, device=device)
    return (predicted - expected) / scale


def _losses(model, config, step, device, dtype) -> dict[str, Tensor]:
    progress = (step - 1) / max(config.steps - 1, 1)
    max_time = 0.2 * min(1.0, 0.25 + 1.25 * progress)
    x = _uniform(config.batch_interior, 0.0, 1.0, device, dtype)
    t = _uniform(config.batch_interior, 0.0, max_time, device, dtype)
    points = torch.cat((x, t), dim=-1)
    if isinstance(model, EulerWaveExplicitAnsatz):
        with torch.no_grad():
            contact = model.contact_position(t)
            shock = model.shock_position(t)
            exclusion = max(4.0 * max(model.contact_width, model.shock_width), 0.012)
            mask = ((torch.abs(x - contact) >= exclusion) & (torch.abs(x - shock) >= exclusion)).reshape(-1)
        points = points[mask]
    residual = euler_strong_residual(
        model,
        points,
        artificial_viscosity=(
            config.artificial_viscosity if config.variant == "av_pinn" else 0.0
        ),
    )
    component_scale = torch.tensor([1.0, 1.0, 2.5], dtype=dtype, device=device)
    result = {
        "pde": torch.mean((residual / component_scale) ** 2),
        "initial": _initial_loss(model, config, device, dtype),
        "boundary": _boundary_loss(model, config, max_time, device, dtype),
    }
    if config.variant in {"ci_pinn", "cw_dkan", "cse_dkan"}:
        cv, entropy = euler_control_volume_and_entropy(
            model, _volumes(config, max_time, device, dtype), config.quadrature_order
        )
        result.update(
            {
                "cv": torch.mean((cv / component_scale) ** 2),
                "entropy": torch.mean(entropy**2),
                "global": torch.mean(_global_balance(model, max_time, device, dtype) ** 2),
            }
        )
    if isinstance(model, EulerWaveExplicitAnsatz):
        manifold_times = _uniform(config.batch_boundary, 0.01, max_time, device, dtype).requires_grad_(True)
        contact_left, contact_right, shock_left, shock_right = model.wave_traces(manifold_times)
        contact_speed = gradient(model.contact_position(manifold_times), manifold_times)
        shock_speed = gradient(model.shock_position(manifold_times), manifold_times)
        contact_conservative_left = primitive_to_conservative(contact_left)
        contact_conservative_right = primitive_to_conservative(contact_right)
        shock_conservative_left = primitive_to_conservative(shock_left)
        shock_conservative_right = primitive_to_conservative(shock_right)
        contact_rh = (
            euler_flux(contact_conservative_right)
            - euler_flux(contact_conservative_left)
            - contact_speed * (contact_conservative_right - contact_conservative_left)
        )
        shock_rh = (
            euler_flux(shock_conservative_right)
            - euler_flux(shock_conservative_left)
            - shock_speed * (shock_conservative_right - shock_conservative_left)
        )
        contact_continuity = F.mse_loss(contact_left[:, 1:], contact_right[:, 1:])
        separation_target = 0.25 * manifold_times
        separation = F.relu(
            model.contact_position(manifold_times) + separation_target - model.shock_position(manifold_times)
        )
        exact = ExactSodSolver()
        target_contact_left = torch.tensor(
            [
                exact.left_star_state.density,
                exact.left_star_state.velocity,
                exact.left_star_state.pressure,
            ],
            dtype=dtype,
            device=device,
        ).expand_as(contact_left)
        target_contact_right = torch.tensor(
            [
                exact.right_star_state.density,
                exact.right_star_state.velocity,
                exact.right_star_state.pressure,
            ],
            dtype=dtype,
            device=device,
        ).expand_as(contact_right)
        target_shock_right = torch.tensor(
            [exact.right.density, exact.right.velocity, exact.right.pressure],
            dtype=dtype,
            device=device,
        ).expand_as(shock_right)
        trace = (
            F.mse_loss(contact_left, target_contact_left)
            + F.mse_loss(contact_right, target_contact_right)
            + F.mse_loss(shock_left, target_contact_right)
            + F.mse_loss(shock_right, target_shock_right)
        )
        result.update(
            {
                "rh": torch.mean(contact_rh**2) + torch.mean(shock_rh**2),
                "contact": contact_continuity,
                "shock_entropy": torch.mean(
                    euler_physical_entropy_violation(shock_left, shock_right) ** 2
                ),
                "separation": torch.mean(separation**2),
                "trace": trace,
            }
        )
    return result


def _total(losses: dict[str, Tensor], config: EulerTrainingConfig) -> Tensor:
    weights = {
        "pde": config.weight_pde,
        "initial": config.weight_initial,
        "boundary": config.weight_boundary,
        "cv": config.weight_cv,
        "entropy": config.weight_entropy,
        "global": config.weight_global,
        "rh": config.weight_rh,
        "contact": config.weight_contact,
        "shock_entropy": config.weight_shock_entropy,
        "separation": config.weight_separation,
        "trace": config.weight_trace,
    }
    return sum(weights[name] * value for name, value in losses.items())


def _gradient_balanced_total(
    losses: dict[str, Tensor],
    model: nn.Module,
    config: EulerTrainingConfig,
    state: dict[str, float],
) -> tuple[Tensor, dict[str, float]]:
    base = {
        "pde": config.weight_pde,
        "initial": config.weight_initial,
        "boundary": config.weight_boundary,
        "cv": config.weight_cv,
        "entropy": config.weight_entropy,
        "global": config.weight_global,
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    terms = {name: base[name] * value for name, value in losses.items()}
    norms = {}
    for name, term in terms.items():
        gradients = torch.autograd.grad(
            term, parameters, retain_graph=True, allow_unused=True
        )
        squared = sum(
            torch.sum(gradient.detach() ** 2)
            for gradient in gradients
            if gradient is not None
        )
        norms[name] = torch.sqrt(squared + torch.finfo(term.dtype).eps)
    target = torch.mean(torch.stack(tuple(norms.values())))
    updated = {}
    for name, norm in norms.items():
        instantaneous = float(
            torch.clamp(
                target / norm,
                config.gradient_weight_minimum,
                config.gradient_weight_maximum,
            ).cpu()
        )
        updated[name] = (
            config.gradient_weight_momentum * state.get(name, 1.0)
            + (1.0 - config.gradient_weight_momentum) * instantaneous
        )
    return sum(updated[name] * term for name, term in terms.items()), updated


@torch.no_grad()
def evaluate_euler(model, config, device, dtype) -> dict[str, float]:
    exact = ExactSodSolver()
    x = np.linspace(0.0, 1.0, config.evaluation_points, endpoint=False) + 0.5 / config.evaluation_points
    points = torch.tensor(np.column_stack((x, np.full_like(x, 0.2))), dtype=dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _ = model(points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference = time.perf_counter() - started
    prediction_np = prediction.cpu().numpy()
    reference = np.column_stack(exact.sample(x, 0.2))
    gamma = exact.gamma
    sound_right = math.sqrt(gamma * exact.right.pressure / exact.right.density)
    shock_speed = exact.right.velocity + sound_right * math.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * exact.p_star / exact.right.pressure
        + (gamma - 1.0) / (2.0 * gamma)
    )
    shock = exact.discontinuity + shock_speed * 0.2
    contact = exact.discontinuity + exact.u_star * 0.2

    def width_at(location, window):
        left_state = float(np.mean(reference[(x > location - 0.025) & (x < location - 0.01), 0]))
        right_state = float(np.mean(reference[(x > location + 0.01) & (x < location + 0.025), 0]))
        try:
            return shock_width_10_90(x, prediction_np[:, 0], left_state, right_state, window)
        except ValueError:
            return 1.0

    shock_location = shock_location_from_gradient(x, prediction_np[:, 0], (0.75, 0.95))
    contact_location = shock_location_from_gradient(x, prediction_np[:, 0], (0.60, 0.75))
    grid = torch.linspace(0.0, 1.0, 1024, device=device, dtype=dtype).reshape(1, -1, 1)
    times = torch.linspace(0.0, 0.2, 9, device=device, dtype=dtype).reshape(-1, 1, 1)
    eval_points = torch.cat((grid.expand(9, -1, -1), times.expand(-1, grid.shape[1], -1)), dim=-1)
    conservative = primitive_to_conservative(model(eval_points.reshape(-1, 2))).reshape(9, grid.shape[1], 3)
    totals = conservative.mean(dim=1)
    initial = primitive_to_conservative(_sod_initial(grid.reshape(-1, 1))).mean(dim=0)
    expected = initial.reshape(1, 3) + times.reshape(-1, 1) * torch.tensor(
        [[0.0, 0.9, 0.0]], dtype=dtype, device=device
    )
    balance = torch.abs(totals - expected).cpu().numpy()
    shock_left = torch.tensor(prediction_np[np.argmin(np.abs(x - (shock - 0.01)))], dtype=dtype)
    shock_right = torch.tensor(prediction_np[np.argmin(np.abs(x - (shock + 0.01)))], dtype=dtype)
    return {
        "density_l1": normalized_lp(prediction_np[:, 0], reference[:, 0], 1),
        "density_l2": normalized_lp(prediction_np[:, 0], reference[:, 0], 2),
        "velocity_l2": normalized_lp(prediction_np[:, 1], reference[:, 1], 2),
        "pressure_l2": normalized_lp(prediction_np[:, 2], reference[:, 2], 2),
        "shock_location_error": abs(shock_location - shock),
        "contact_location_error": abs(contact_location - contact),
        "shock_width_10_90": width_at(shock, (0.75, 0.95)),
        "contact_width_10_90": width_at(contact, (0.60, 0.75)),
        "max_mass_balance_error": float(np.max(balance[:, 0])),
        "max_momentum_balance_error": float(np.max(balance[:, 1])),
        "max_energy_balance_error": float(np.max(balance[:, 2])),
        "shock_entropy_violation": float(
            euler_physical_entropy_violation(shock_left.reshape(1, 3), shock_right.reshape(1, 3)).item()
        ),
        "minimum_density": float(np.min(prediction_np[:, 0])),
        "minimum_pressure": float(np.min(prediction_np[:, 2])),
        "inference_seconds": inference,
    }


def train_euler(config: EulerTrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    _seed(config.seed)
    dtype = _dtype(config.dtype)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_euler_model(config.variant, device, dtype)
    optimized_parameter_count = trainable_parameter_count(model)
    if isinstance(model, EulerWaveExplicitAnsatz) and config.freeze_wave_geometry:
        model.contact_coefficients.requires_grad_(False)
        model.shock_coefficients.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    gradient_weight_state: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        progress = (step - 1) / max(config.steps - 1, 1)
        learning_rate = config.min_learning_rate + 0.5 * (
            config.learning_rate - config.min_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if isinstance(model, EulerWaveExplicitAnsatz):
            anneal = min(1.0, max(0.0, (progress - 0.20) / 0.65))
            contact_width = 0.025 * (0.004 / 0.025) ** anneal
            shock_width = 0.020 * (0.0025 / 0.020) ** anneal
            model.set_gate_widths(contact_width, shock_width)
        optimizer.zero_grad(set_to_none=True)
        losses = _losses(model, config, step, device, dtype)
        if config.variant == "gw_pinn":
            total, gradient_weight_state = _gradient_balanced_total(
                losses, model, config, gradient_weight_state
            )
        else:
            total = _total(losses, config)
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite Euler loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {"step": step, "total": float(total.detach()), "learning_rate": learning_rate}
            row.update({name: float(value.detach()) for name, value in losses.items()})
            row.update(
                {
                    f"gradient_weight_{name}": value
                    for name, value in gradient_weight_state.items()
                }
            )
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metrics = evaluate_euler(model, config, device, dtype)
    if isinstance(model, EulerWaveExplicitAnsatz):
        final_time = torch.tensor([[0.2]], dtype=dtype, device=device)
        metrics.update(
            {
                "learned_contact_position": float(model.contact_position(final_time).detach()),
                "learned_shock_position": float(model.shock_position(final_time).detach()),
                "final_contact_width": model.contact_width,
                "final_shock_width": model.shock_width,
            }
        )
    metrics.update(
        {
            "training_seconds": training_seconds,
            "peak_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
            "trainable_parameters": float(trainable_parameter_count(model)),
            "optimized_parameters": float(optimized_parameter_count),
        }
    )
    payload = {
        "config": asdict(config),
        "architecture_id": model.__class__.__name__,
        "history": history,
        "metrics": metrics,
        "environment": {"torch": torch.__version__, "device": str(device)},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "architecture_id": model.__class__.__name__,
        },
        output / "model.pt",
    )
    return payload
    EulerWaveExplicitAnsatz,
