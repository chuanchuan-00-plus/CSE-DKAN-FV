"""Reproducible development training loop for the 1D Burgers MVP."""

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

from .benchmarks import PeriodicBurgersProblem
from .losses import (
    RectangularControlVolumes,
    burgers_control_volume_balance,
    burgers_control_volume_and_entropy,
    burgers_entropy_violation,
    burgers_viscous_entropy_violation,
    burgers_periodic_global_conservation,
    burgers_rankine_hugoniot_residual,
    burgers_shock_entropy_violation,
    burgers_strong_residual,
    gradient,
)
from .metrics import (
    gradient_equivalent_width,
    normalized_lp,
    overshoot_undershoot,
    periodic_mass,
    shock_location_from_gradient,
    shock_width_10_90,
    total_variation_excess,
)
from .models import (
    BurgersInitialConditionAnsatz,
    DKAN,
    ConservativeSpectralShockBurgersAnsatz,
    LowRankShockBurgersAnsatz,
    MLP,
    PeriodicCoordinateMap,
    ShockExplicitBurgersAnsatz,
    ViscousLayerConservativeSpectralBurgersAnsatz,
    trainable_parameter_count,
)


@dataclass
class BurgersTrainingConfig:
    variant: str = "mlp_pinn"
    seed: int = 0
    steps: int = 2000
    batch_interior: int = 1024
    batch_boundary: int = 128
    batch_control_volumes: int = 128
    batch_rh: int = 128
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 5.0e-5
    weight_pde: float = 1.0
    weight_boundary: float = 10.0
    weight_cv: float = 100.0
    weight_entropy: float = 50.0
    weight_global: float = 200.0
    weight_interface_cv: float = 0.0
    weight_interface_entropy: float = 0.0
    weight_rh: float = 5.0
    weight_shock_entropy: float = 1.0
    weight_characteristic_position: float = 100.0
    weight_characteristic_trace: float = 30.0
    weight_maximum_principle: float = 1000.0
    quadrature_order: int = 4
    dtype: str = "float32"
    device: str = "cuda"
    log_every: int = 100
    reference_cells: int = 4096
    evaluation_points: int = 2048
    gate_width_start: float = 0.06
    gate_width_end: float = 0.003
    shock_exclusion_factor: float = 3.0
    conservative_stride: int = 2
    interface_pretrain_steps: int = 500
    interface_pretrain_learning_rate: float = 5.0e-3
    viscosity: float = 0.0
    amplitude: float = 1.0
    mean: float = 0.2
    final_time: float = 0.6
    artificial_viscosity: float = 0.01
    gradient_weight_momentum: float = 0.9
    gradient_weight_minimum: float = 0.1
    gradient_weight_maximum: float = 10.0

    @classmethod
    def from_json(cls, path: str | Path) -> "BurgersTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def _dtype(name: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if name not in choices:
        raise ValueError(f"unsupported dtype: {name}")
    return choices[name]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_burgers_model(
    variant: str,
    problem: PeriodicBurgersProblem,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    coordinate_map = PeriodicCoordinateMap(problem.x_left, problem.x_right, problem.final_time)
    if variant in {"mlp_pinn", "gw_pinn", "av_pinn", "ci_pinn"}:
        correction = MLP([3, 64, 64, 64, 1])
        model: nn.Module = BurgersInitialConditionAnsatz(
            correction, coordinate_map, problem.mean, problem.amplitude, problem.origin
        )
    elif variant in {"dkan_pinn", "cw_dkan"}:
        correction = DKAN([3, 25, 25, 1], embedding_frequencies=12)
        model = BurgersInitialConditionAnsatz(
            correction, coordinate_map, problem.mean, problem.amplitude, problem.origin
        )
    elif variant == "cse_dkan_two_expert":
        left = DKAN([3, 17, 17, 1], embedding_frequencies=12)
        right = DKAN([3, 17, 17, 1], embedding_frequencies=12)
        model = ShockExplicitBurgersAnsatz(
            left,
            right,
            coordinate_map,
            final_time=problem.final_time,
            shock_origin=problem.shock_origin,
            initial_shock_speed=problem.initial_shock_speed,
            initial_onset_time=problem.shock_formation_time,
            mean=problem.mean,
            amplitude=problem.amplitude,
            origin=problem.origin,
        )
    elif variant == "cse_dkan":
        basis = DKAN([1, 18, 18, 21], embedding_frequencies=8)
        model = ConservativeSpectralShockBurgersAnsatz(
            basis,
            coordinate_map,
            final_time=problem.final_time,
            shock_origin=problem.shock_origin,
            initial_shock_speed=problem.initial_shock_speed,
            initial_onset_time=problem.shock_formation_time,
            mean=problem.mean,
            amplitude=problem.amplitude,
            origin=problem.origin,
            modes=10,
        )
    elif variant == "cse_dkan_viscous":
        if problem.viscosity <= 0.0:
            raise ValueError("cse_dkan_viscous requires positive viscosity")
        basis = DKAN([1, 18, 18, 21], embedding_frequencies=8)
        model = ViscousLayerConservativeSpectralBurgersAnsatz(
            basis,
            coordinate_map,
            final_time=problem.final_time,
            shock_origin=problem.shock_origin,
            initial_shock_speed=problem.initial_shock_speed,
            initial_onset_time=problem.shock_formation_time,
            mean=problem.mean,
            amplitude=problem.amplitude,
            origin=problem.origin,
            modes=10,
            viscosity=problem.viscosity,
        )
    elif variant == "cse_dkan_pointwise":
        basis = DKAN([3, 25, 25, 2], embedding_frequencies=12)
        model = LowRankShockBurgersAnsatz(
            basis,
            coordinate_map,
            final_time=problem.final_time,
            shock_origin=problem.shock_origin,
            initial_shock_speed=problem.initial_shock_speed,
            initial_onset_time=problem.shock_formation_time,
            mean=problem.mean,
            amplitude=problem.amplitude,
            origin=problem.origin,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    return model.to(device=device, dtype=dtype)


def _uniform(
    count: int,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return low + (high - low) * torch.rand((count, 1), device=device, dtype=dtype)


def _interior_points(
    config: BurgersTrainingConfig,
    problem: PeriodicBurgersProblem,
    device,
    dtype,
    model: nn.Module | None = None,
) -> Tensor:
    x = _uniform(config.batch_interior, problem.x_left, problem.x_right, device, dtype)
    t = _uniform(config.batch_interior, 0.0, problem.final_time, device, dtype)
    if isinstance(model, ViscousLayerConservativeSpectralBurgersAnsatz) and config.batch_interior >= 4:
        # Resolve the O(nu / jump) layer explicitly; uniform sampling almost
        # never sees it when nu is 1e-3 or smaller.
        layer_count = config.batch_interior // 2
        layer_t = _uniform(
            layer_count,
            problem.shock_formation_time,
            problem.final_time,
            device,
            dtype,
        )
        with torch.no_grad():
            position = model.shock_position(layer_t)
            width = model.physical_gate_width(layer_t)
        normalized_offset = _uniform(layer_count, -6.0, 6.0, device, dtype)
        x[:layer_count] = position + normalized_offset * width
        t[:layer_count] = layer_t
    return torch.cat((x, t), dim=-1)


def _periodic_boundary_loss(model: nn.Module, config, problem, device, dtype) -> Tensor:
    times = _uniform(config.batch_boundary, 0.0, problem.final_time, device, dtype)
    left = torch.cat((torch.full_like(times, problem.x_left), times), dim=-1).requires_grad_(True)
    right = torch.cat((torch.full_like(times, problem.x_right), times), dim=-1).requires_grad_(True)
    u_left, u_right = model(left), model(right)
    du_left = gradient(u_left, left)[:, :1]
    du_right = gradient(u_right, right)[:, :1]
    return F.mse_loss(u_left, u_right) + 0.1 * F.mse_loss(du_left, du_right)


def _control_volumes(config, problem, device, dtype, model=None) -> RectangularControlVolumes:
    n = config.batch_control_volumes
    hx = _uniform(n, 0.01, 0.08, device, dtype)
    ht = _uniform(n, 0.005, 0.035, device, dtype)
    x = problem.x_left + hx + (problem.x_right - problem.x_left - 2.0 * hx) * torch.rand_like(hx)
    t = ht + (problem.final_time - 2.0 * ht) * torch.rand_like(ht)
    if isinstance(model, (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz)) and n >= 2:
        # Half of the volumes deliberately straddle the learned manifold; the
        # remainder retain global coverage and prevent interface-only fitting.
        interface_count = n // 2
        interface_hx = _uniform(interface_count, 0.01, 0.08, device, dtype)
        interface_ht = _uniform(interface_count, 0.005, 0.02, device, dtype)
        available_time = problem.final_time - problem.shock_formation_time - 2.0 * interface_ht
        interface_t = (
            problem.shock_formation_time
            + interface_ht
            + available_time * torch.rand_like(interface_ht)
        )
        with torch.no_grad():
            interface_x = model.shock_position(interface_t)
        jitter = 0.25 * interface_hx * (2.0 * torch.rand_like(interface_x) - 1.0)
        x[:interface_count] = interface_x + jitter
        t[:interface_count] = interface_t
        hx[:interface_count] = interface_hx
        ht[:interface_count] = interface_ht
    return RectangularControlVolumes(torch.cat((x, t), dim=-1), hx, ht)


def _conservative_losses(model, config, problem, device, dtype) -> dict[str, Tensor]:
    volumes = _control_volumes(config, problem, device, dtype, model)
    if problem.viscosity == 0.0:
        cv, entropy = burgers_control_volume_and_entropy(
            model, volumes, config.quadrature_order, normalization="sqrt_area"
        )
    else:
        cv = burgers_control_volume_balance(model, volumes, config.quadrature_order, problem.viscosity)
        entropy = burgers_viscous_entropy_violation(
            model, volumes, problem.viscosity, config.quadrature_order
        )
        # Convert the area-normalized diagnostic to the weak energy norm used
        # for training; independent evaluation remains area-normalized.
        area_scale = torch.sqrt(
            4.0
            * volumes.half_width_x.reshape(-1, 1)
            * volumes.half_width_t.reshape(-1, 1)
        )
        cv = cv * area_scale
        entropy = entropy * area_scale
    x = torch.linspace(problem.x_left, problem.x_right, 128, device=device, dtype=dtype)[:-1]
    times = torch.linspace(0.0, problem.final_time, 5, device=device, dtype=dtype)[1:].reshape(-1, 1)
    global_drift = burgers_periodic_global_conservation(model, x, times, problem.initial_torch)
    return {
        "cv": torch.mean(cv**2),
        "entropy": torch.mean(entropy**2),
        "global": torch.mean(global_drift**2),
        **(
            {
                "interface_cv": torch.mean(
                    (cv[: config.batch_control_volumes // 2] / torch.sqrt(
                        4.0
                        * volumes.half_width_x[: config.batch_control_volumes // 2].reshape(-1, 1)
                        * volumes.half_width_t[: config.batch_control_volumes // 2].reshape(-1, 1)
                    ))
                    ** 2
                ),
                "interface_entropy": torch.mean(
                    (entropy[: config.batch_control_volumes // 2] / torch.sqrt(
                        4.0
                        * volumes.half_width_x[: config.batch_control_volumes // 2].reshape(-1, 1)
                        * volumes.half_width_t[: config.batch_control_volumes // 2].reshape(-1, 1)
                    ))
                    ** 2
                ),
            }
            if isinstance(model, (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz))
            else {}
        ),
    }


def _shock_losses(model, config, problem, device, dtype) -> dict[str, Tensor]:
    start = min(problem.shock_formation_time + 0.02, 0.95 * problem.final_time)
    times = _uniform(config.batch_rh, start, problem.final_time, device, dtype)
    offset = max(4.0 * model.gate_width, 0.012)
    rh = burgers_rankine_hugoniot_residual(model, model.shock_position, times, offset)
    entropy = burgers_shock_entropy_violation(model, model.shock_position, times, offset)
    return {"rh": torch.mean(rh**2), "shock_entropy": torch.mean(entropy**2)}


def _low_rank_shock_losses(
    model: LowRankShockBurgersAnsatz, config, problem, device, dtype
) -> dict[str, Tensor]:
    """Close the zero-jump RH degeneracy with two characteristic traces."""
    start = min(problem.shock_formation_time + 0.02, 0.95 * problem.final_time)
    times = _uniform(config.batch_rh, start, problem.final_time, device, dtype).requires_grad_(True)
    position = model.shock_position(times)
    speed = gradient(position, times)
    offset = max(3.0 * model.gate_width, 0.012)
    trace_points = torch.cat(
        (
            torch.cat((position - offset, times), dim=-1),
            torch.cat((position + offset, times), dim=-1),
        ),
        dim=0,
    )
    trace_states = model(trace_points)
    u_left, u_right = torch.split(trace_states, times.shape[0], dim=0)
    rh = 0.5 * (u_right**2 - u_left**2) - speed * (u_right - u_left)
    entropy = torch.cat(
        (
            F.relu(u_right - u_left),
            F.relu(speed - u_left),
            F.relu(u_right - speed),
        ),
        dim=-1,
    )
    foot_left, foot_right = model.characteristic_footpoints(times)
    initial_left = problem.initial_torch(foot_left)
    initial_right = problem.initial_torch(foot_right)
    characteristic_left = foot_left + times * initial_left
    characteristic_right = foot_right + times * initial_right
    position_residual = torch.cat(
        (position - characteristic_left, position - characteristic_right), dim=-1
    )
    trace_residual = torch.cat((u_left - initial_left, u_right - initial_right), dim=-1)
    return {
        "rh": torch.mean(rh**2),
        "shock_entropy": torch.mean(entropy**2),
        "characteristic_position": torch.mean(position_residual**2),
        "characteristic_trace": torch.mean(trace_residual**2),
    }


def _viscous_layer_trace_losses(
    model: ViscousLayerConservativeSpectralBurgersAnsatz,
    config: BurgersTrainingConfig,
    problem: PeriodicBurgersProblem,
    device,
    dtype,
) -> dict[str, Tensor]:
    """Match the small-viscosity outer states without imposing a discontinuity.

    The characteristic construction supplies asymptotic states on either side
    of the smooth layer.  Diffusion and the layer profile remain governed by
    the viscous strong residual rather than an inviscid RH penalty.
    """
    start = min(problem.shock_formation_time + 0.02, 0.95 * problem.final_time)
    times = _uniform(config.batch_rh, start, problem.final_time, device, dtype)
    position = model.shock_position(times)
    foot_left, foot_right = model.characteristic_footpoints(times)
    initial_left = problem.initial_torch(foot_left)
    initial_right = problem.initial_torch(foot_right)
    characteristic_left = foot_left + times * initial_left
    characteristic_right = foot_right + times * initial_right
    state_left, state_right = model.interface_states(times)
    return {
        "characteristic_position": torch.mean(
            torch.cat(
                (position - characteristic_left, position - characteristic_right), dim=-1
            )
            ** 2
        ),
        "characteristic_trace": torch.mean(
            torch.cat((state_left - initial_left, state_right - initial_right), dim=-1) ** 2
        ),
    }


def _physics_loss(model, config, problem, device, dtype, step: int = 1) -> dict[str, Tensor]:
    points = _interior_points(config, problem, device, dtype, model)
    shock_types = (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz)
    if isinstance(model, shock_types) and problem.viscosity == 0.0:
        with torch.no_grad():
            position = model.shock_position(points[:, 1:2])
            active = points[:, 1:2] >= model.onset_time().detach()
            exclusion_width = min(config.shock_exclusion_factor * model.gate_width, 0.15)
            outside = torch.abs(points[:, :1] - position) >= exclusion_width
            mask = (~active | outside).reshape(-1)
        points = points[mask]
    residual_viscosity = (
        max(problem.viscosity, config.artificial_viscosity)
        if config.variant == "av_pinn"
        else problem.viscosity
    )
    residual = burgers_strong_residual(model, points, residual_viscosity)
    losses = {
        "pde": torch.mean(residual**2),
        "boundary": _periodic_boundary_loss(model, config, problem, device, dtype),
    }
    if isinstance(model, ViscousLayerConservativeSpectralBurgersAnsatz):
        lower = problem.mean - abs(problem.amplitude)
        upper = problem.mean + abs(problem.amplitude)
        values = model(points)
        losses["maximum_principle"] = torch.mean(
            F.relu(lower - values) ** 2 + F.relu(values - upper) ** 2
        )
    conservative_due = (step - 1) % config.conservative_stride == 0
    if config.variant in {
        "cw_dkan",
        "cse_dkan",
        "cse_dkan_viscous",
        "cse_dkan_pointwise",
        "cse_dkan_two_expert",
        "ci_pinn",
    } and conservative_due:
        losses.update(_conservative_losses(model, config, problem, device, dtype))
    if isinstance(model, ViscousLayerConservativeSpectralBurgersAnsatz):
        losses.update(_viscous_layer_trace_losses(model, config, problem, device, dtype))
    elif isinstance(model, LowRankShockBurgersAnsatz) and problem.viscosity == 0.0:
        losses.update(_low_rank_shock_losses(model, config, problem, device, dtype))
    elif isinstance(model, ShockExplicitBurgersAnsatz):
        losses.update(_shock_losses(model, config, problem, device, dtype))
    return losses


def _weighted_total(
    losses: dict[str, Tensor], config: BurgersTrainingConfig, progress: float = 1.0
) -> Tensor:
    shock_ramp = min(1.0, max(0.0, (progress - 0.15) / 0.35))
    weights = {
        "pde": config.weight_pde,
        "boundary": config.weight_boundary,
        "cv": config.weight_cv,
        "entropy": config.weight_entropy,
        "global": config.weight_global,
        "interface_cv": config.weight_interface_cv,
        "interface_entropy": config.weight_interface_entropy,
        "rh": config.weight_rh * shock_ramp,
        "shock_entropy": config.weight_shock_entropy * shock_ramp,
        "characteristic_position": config.weight_characteristic_position * shock_ramp,
        "characteristic_trace": config.weight_characteristic_trace * shock_ramp,
        "maximum_principle": config.weight_maximum_principle,
    }
    return sum(weights[name] * value for name, value in losses.items())


def _gradient_balanced_total(
    losses: dict[str, Tensor],
    model: nn.Module,
    config: BurgersTrainingConfig,
    state: dict[str, float],
) -> tuple[Tensor, dict[str, float]]:
    """Balance fixed loss terms by their parameter-gradient norms."""
    base = {
        "pde": config.weight_pde,
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
        norms[name] = torch.sqrt(
            squared + torch.finfo(term.dtype).eps
        )
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
        previous = state.get(name, 1.0)
        updated[name] = (
            config.gradient_weight_momentum * previous
            + (1.0 - config.gradient_weight_momentum) * instantaneous
        )
    return sum(updated[name] * term for name, term in terms.items()), updated


def _learning_rate(config: BurgersTrainingConfig, progress: float, shock_explicit: bool) -> float:
    if not shock_explicit:
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine
    if progress < 0.20:
        return config.learning_rate
    if progress < 0.55:
        return 0.70 * config.learning_rate
    if progress < 0.80:
        return 0.40 * config.learning_rate
    tail = (progress - 0.80) / 0.20
    cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
    return config.min_learning_rate + (0.40 * config.learning_rate - config.min_learning_rate) * cosine


def _pretrain_interface_geometry(
    model: LowRankShockBurgersAnsatz,
    problem: PeriodicBurgersProblem,
    config: BurgersTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Solve the characteristic/RH geometry subproblem before field fitting."""
    model.raw_onset.requires_grad_(False)
    geometry_parameters = [
        model.trajectory.coefficients,
        model.footpoint_center_coefficients,
        model.footpoint_separation_coefficients,
    ]
    optimizer = torch.optim.Adam(geometry_parameters, lr=config.interface_pretrain_learning_rate)
    last_position = torch.tensor(float("inf"), device=device, dtype=dtype)
    last_rh = torch.tensor(float("inf"), device=device, dtype=dtype)
    for _ in range(config.interface_pretrain_steps):
        times = torch.linspace(
            problem.shock_formation_time + 0.005,
            problem.final_time,
            192,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1).requires_grad_(True)
        position = model.shock_position(times)
        speed = gradient(position, times)
        foot_left, foot_right = model.characteristic_footpoints(times)
        state_left = problem.initial_torch(foot_left)
        state_right = problem.initial_torch(foot_right)
        position_residual = torch.cat(
            (
                position - (foot_left + times * state_left),
                position - (foot_right + times * state_right),
            ),
            dim=-1,
        )
        rh = 0.5 * (state_right**2 - state_left**2) - speed * (state_right - state_left)
        last_position = torch.mean(position_residual**2)
        last_rh = torch.mean(rh**2)
        loss = 100.0 * last_position + 10.0 * last_rh
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for parameter in geometry_parameters:
        parameter.requires_grad_(False)
    return {
        "interface_position_mse": float(last_position.detach()),
        "interface_rh_mse": float(last_rh.detach()),
    }


@torch.no_grad()
def evaluate_burgers(
    model: nn.Module,
    problem: PeriodicBurgersProblem,
    reference_x: np.ndarray,
    reference_u: np.ndarray,
    evaluation_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    x = np.linspace(problem.x_left, problem.x_right, evaluation_points, endpoint=False)
    points = torch.as_tensor(
        np.column_stack((x, np.full_like(x, problem.final_time))), dtype=dtype, device=device
    )
    # Warm up lazy CUDA kernels before the fixed-grid timing measurement.
    _ = model(points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = model(points).detach().cpu().numpy().reshape(-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start
    reference = np.interp(x, reference_x, reference_u, period=problem.x_right - problem.x_left)
    expected_shock = problem.origin + problem.mean * problem.final_time
    window = (expected_shock - 0.15, expected_shock + 0.15)
    left_mask = (x >= expected_shock - 0.035) & (x <= expected_shock - 0.012)
    right_mask = (x >= expected_shock + 0.012) & (x <= expected_shock + 0.035)
    left_state = float(np.mean(reference[left_mask]))
    right_state = float(np.mean(reference[right_mask]))
    shock_location = shock_location_from_gradient(x, prediction, window)
    reference_location = shock_location_from_gradient(x, reference, window)
    try:
        width_10_90 = shock_width_10_90(x, prediction, left_state, right_state, window)
        shock_width_failure = 0.0
    except ValueError:
        width_10_90 = problem.x_right - problem.x_left
        shock_width_failure = 1.0
    reference_width = shock_width_10_90(x, reference, left_state, right_state, window)

    # Independent deterministic space-time control volumes for evaluation.
    grid_x = torch.linspace(problem.x_left + 0.04, problem.x_right - 0.04, 8, device=device, dtype=dtype)
    grid_t = torch.linspace(0.02, problem.final_time - 0.02, 8, device=device, dtype=dtype)
    mesh_x, mesh_t = torch.meshgrid(grid_x, grid_t, indexing="ij")
    centers = torch.stack((mesh_x.reshape(-1), mesh_t.reshape(-1)), dim=-1)
    hx = torch.full((centers.shape[0], 1), 0.04, device=device, dtype=dtype)
    ht = torch.full((centers.shape[0], 1), 0.015, device=device, dtype=dtype)
    volumes = RectangularControlVolumes(centers, hx, ht)
    if problem.viscosity == 0.0:
        cv, entropy = burgers_control_volume_and_entropy(model, volumes, quadrature_order=6)
    else:
        cv = burgers_control_volume_balance(model, volumes, quadrature_order=6, viscosity=problem.viscosity)
        entropy = burgers_viscous_entropy_violation(
            model, volumes, problem.viscosity, quadrature_order=6
        )

    shock_times = torch.linspace(
        problem.shock_formation_time + 0.03,
        problem.final_time - 0.02,
        16,
        device=device,
        dtype=dtype,
    ).reshape(-1, 1)
    shock_centers = torch.cat((problem.origin + problem.mean * shock_times, shock_times), dim=-1)
    shock_hx = torch.full((16, 1), 0.045, device=device, dtype=dtype)
    shock_ht = torch.full((16, 1), 0.012, device=device, dtype=dtype)
    shock_volumes = RectangularControlVolumes(shock_centers, shock_hx, shock_ht)
    if problem.viscosity == 0.0:
        shock_cv, shock_entropy = burgers_control_volume_and_entropy(
            model, shock_volumes, quadrature_order=8
        )
    else:
        shock_cv = burgers_control_volume_balance(
            model, shock_volumes, quadrature_order=8, viscosity=problem.viscosity
        )
        shock_entropy = burgers_viscous_entropy_violation(
            model, shock_volumes, problem.viscosity, quadrature_order=8
        )

    mass_times = torch.linspace(0.0, problem.final_time, 9, device=device, dtype=dtype)
    mass_x = torch.linspace(problem.x_left, problem.x_right, 512, device=device, dtype=dtype)[:-1]
    drift = burgers_periodic_global_conservation(model, mass_x, mass_times, problem.initial_torch)
    return {
        "normalized_l1": normalized_lp(prediction, reference, 1),
        "normalized_l2": normalized_lp(prediction, reference, 2),
        "normalized_linf": normalized_lp(prediction, reference, np.inf),
        "mass_error": abs(
            periodic_mass(prediction, problem.x_right - problem.x_left)
            - periodic_mass(reference, problem.x_right - problem.x_left)
        ),
        "tv_excess": total_variation_excess(prediction, reference),
        "overshoot_undershoot": overshoot_undershoot(prediction, reference),
        "shock_location_error": abs(shock_location - reference_location),
        "shock_width_10_90": width_10_90,
        "shock_width_failure": shock_width_failure,
        "reference_shock_width_10_90": reference_width,
        "shock_width_absolute_error": abs(width_10_90 - reference_width),
        "gradient_equivalent_width": gradient_equivalent_width(
            x, prediction, left_state, right_state, window
        ),
        "reference_gradient_equivalent_width": gradient_equivalent_width(
            x, reference, left_state, right_state, window
        ),
        "cv_balance_rmse": float(torch.sqrt(torch.mean(cv**2)).detach().cpu()),
        "shock_cv_balance_rmse": float(torch.sqrt(torch.mean(shock_cv**2)).cpu()),
        "entropy_violation_mean": float(torch.mean(entropy).detach().cpu()),
        "entropy_violation_max": float(torch.max(entropy).detach().cpu()),
        "entropy_violation_fraction": float(
            torch.mean((entropy > 1.0e-6).to(dtype)).detach().cpu()
        ),
        "shock_entropy_violation_mean": float(torch.mean(shock_entropy).cpu()),
        "max_mass_drift": float(torch.max(torch.abs(drift)).cpu()),
        "inference_seconds": inference_seconds,
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def train_burgers(config: BurgersTrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    _seed_everything(config.seed)
    dtype = _dtype(config.dtype)
    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    problem = PeriodicBurgersProblem(
        final_time=config.final_time,
        mean=config.mean,
        amplitude=config.amplitude,
        viscosity=config.viscosity,
    )
    model = build_burgers_model(config.variant, problem, device, dtype)
    optimized_parameter_count = trainable_parameter_count(model)
    interface_pretrain_metrics: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if isinstance(model, LowRankShockBurgersAnsatz) and config.interface_pretrain_steps > 0:
        interface_pretrain_metrics = _pretrain_interface_geometry(model, problem, config, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    gradient_weight_state: dict[str, float] = {}
    for step in range(1, config.steps + 1):
        progress = (step - 1) / max(config.steps - 1, 1)
        shock_explicit = isinstance(model, (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz))
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(config, progress, shock_explicit)
        if shock_explicit and not isinstance(model, ViscousLayerConservativeSpectralBurgersAnsatz):
            # Keep the gate broad during smooth pretraining, then sharpen it.
            anneal = max(0.0, (progress - 0.20) / 0.60)
            anneal = min(anneal, 1.0)
            width = config.gate_width_start * (config.gate_width_end / config.gate_width_start) ** anneal
            model.set_gate_width(width)
        optimizer.zero_grad(set_to_none=True)
        losses = _physics_loss(model, config, problem, device, dtype, step)
        if config.variant == "gw_pinn":
            total, gradient_weight_state = _gradient_balanced_total(
                losses, model, config, gradient_weight_state
            )
        else:
            total = _weighted_total(losses, config, progress)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {"step": float(step), "total": float(total.detach())}
            row.update({name: float(value.detach()) for name, value in losses.items()})
            row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            row.update(
                {
                    f"gradient_weight_{name}": value
                    for name, value in gradient_weight_state.items()
                }
            )
            if isinstance(model, (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz)):
                row["gate_width"] = model.gate_width
                row["onset_time"] = float(model.onset_time().detach())
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    reference_x, snapshots = problem.reference(config.reference_cells)
    metrics = evaluate_burgers(
        model,
        problem,
        reference_x,
        snapshots[problem.final_time],
        config.evaluation_points,
        device,
        dtype,
    )
    metrics.update(
        {
            "training_seconds": training_seconds,
            "peak_memory_bytes": float(peak_memory),
            "optimized_parameters": float(optimized_parameter_count),
            "trainable_parameters_after_geometry_freeze": float(trainable_parameter_count(model)),
        }
    )
    metrics.update(interface_pretrain_metrics)
    if isinstance(model, (ShockExplicitBurgersAnsatz, LowRankShockBurgersAnsatz)):
        final_t = torch.tensor([[problem.final_time]], dtype=dtype, device=device)
        metrics["learned_shock_position"] = float(model.shock_position(final_t).detach())
        metrics["learned_onset_time"] = float(model.onset_time().detach())
        metrics["final_gate_width"] = model.gate_width
        if isinstance(model, ViscousLayerConservativeSpectralBurgersAnsatz):
            metrics["final_physical_gate_width"] = float(
                model.physical_gate_width(final_t).detach()
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": asdict(config),
        "problem": asdict(problem),
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "architecture_id": model.__class__.__name__,
        "history": history,
        "metrics": metrics,
    }
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
