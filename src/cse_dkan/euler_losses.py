"""Strong, weak, RH, and entropy diagnostics for one-dimensional Euler PINNs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .euler_finite_volume import euler_flux, primitive_to_conservative
from .losses import (
    RectangularControlVolumes,
    _boundary_samples,
    _integrate_boundary_values,
    gradient,
)


def _conservative_state(model: nn.Module, points: Tensor, gamma: float) -> Tensor:
    return primitive_to_conservative(model(points), gamma)


def euler_strong_residual(
    model: nn.Module,
    points: Tensor,
    gamma: float = 1.4,
    artificial_viscosity: float = 0.0,
) -> Tensor:
    if artificial_viscosity < 0.0:
        raise ValueError("artificial_viscosity must be non-negative")
    points = points.requires_grad_(True)
    conservative = _conservative_state(model, points, gamma)
    flux = euler_flux(conservative, gamma)
    residual_components = []
    for component in range(3):
        state_derivative = gradient(conservative[:, component : component + 1], points)
        flux_derivative = gradient(flux[:, component : component + 1], points)
        residual = state_derivative[:, 1:2] + flux_derivative[:, :1]
        if artificial_viscosity > 0.0:
            state_second_derivative = gradient(
                state_derivative[:, :1], points
            )[:, :1]
            residual = residual - artificial_viscosity * state_second_derivative
        residual_components.append(residual)
    return torch.cat(residual_components, dim=-1)


def euler_control_volume_balance(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    quadrature_order: int = 4,
    gamma: float = 1.4,
) -> Tensor:
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]
    names = ("top", "bottom", "right", "left")
    counts = [samples[name][0].shape[0] for name in names]
    points = torch.cat([samples[name][0] for name in names], dim=0)
    conservative = _conservative_state(model, points, gamma)
    states = torch.split(conservative, counts, dim=0)
    top = _integrate_boundary_values(states[0], samples["top"][1], n)
    bottom = _integrate_boundary_values(states[1], samples["bottom"][1], n)
    right = _integrate_boundary_values(euler_flux(states[2], gamma), samples["right"][1], n)
    left = _integrate_boundary_values(euler_flux(states[3], gamma), samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    return (top - bottom + right - left) / area


def euler_control_volume_and_entropy(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    quadrature_order: int = 4,
    gamma: float = 1.4,
    normalization: str = "sqrt_area",
) -> tuple[Tensor, Tensor]:
    """Fused Euler weak balance and mathematical-entropy violation."""
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]
    names = ("top", "bottom", "right", "left")
    counts = [samples[name][0].shape[0] for name in names]
    points = torch.cat([samples[name][0] for name in names], dim=0)
    primitive_all = model(points)
    primitive = torch.split(primitive_all, counts, dim=0)
    conservative = tuple(primitive_to_conservative(value, gamma) for value in primitive)
    top = _integrate_boundary_values(conservative[0], samples["top"][1], n)
    bottom = _integrate_boundary_values(conservative[1], samples["bottom"][1], n)
    right = _integrate_boundary_values(euler_flux(conservative[2], gamma), samples["right"][1], n)
    left = _integrate_boundary_values(euler_flux(conservative[3], gamma), samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    if normalization == "area":
        normalizer = area
    elif normalization == "sqrt_area":
        normalizer = torch.sqrt(area)
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    balance = (top - bottom + right - left) / normalizer

    def entropy_and_flux(value: Tensor) -> tuple[Tensor, Tensor]:
        density, velocity, pressure = value.unbind(dim=-1)
        physical_entropy = torch.log(pressure) - gamma * torch.log(density)
        mathematical_entropy = -density * physical_entropy / (gamma - 1.0)
        return mathematical_entropy.unsqueeze(-1), (velocity * mathematical_entropy).unsqueeze(-1)

    eta_top, _ = entropy_and_flux(primitive[0])
    eta_bottom, _ = entropy_and_flux(primitive[1])
    _, q_right = entropy_and_flux(primitive[2])
    _, q_left = entropy_and_flux(primitive[3])
    entropy_balance = (
        _integrate_boundary_values(eta_top, samples["top"][1], n)
        - _integrate_boundary_values(eta_bottom, samples["bottom"][1], n)
        + _integrate_boundary_values(q_right, samples["right"][1], n)
        - _integrate_boundary_values(q_left, samples["left"][1], n)
    ) / normalizer
    return balance, F.relu(entropy_balance)


def euler_rankine_hugoniot_residual(
    model: nn.Module,
    shock_position,
    times: Tensor,
    offset: float,
    gamma: float = 1.4,
) -> Tensor:
    times = times.reshape(-1, 1).requires_grad_(True)
    position = shock_position(times)
    speed = gradient(position, times)
    points = torch.cat(
        (
            torch.cat((position - offset, times), dim=-1),
            torch.cat((position + offset, times), dim=-1),
        ),
        dim=0,
    )
    conservative = _conservative_state(model, points, gamma)
    left, right = torch.split(conservative, times.shape[0], dim=0)
    return euler_flux(right, gamma) - euler_flux(left, gamma) - speed * (right - left)


def euler_physical_entropy_violation(
    primitive_left: Tensor,
    primitive_right: Tensor,
    gamma: float = 1.4,
) -> Tensor:
    """Violation for a right-moving compressive shock: s_behind >= s_ahead."""
    density_left, _, pressure_left = primitive_left.unbind(dim=-1)
    density_right, _, pressure_right = primitive_right.unbind(dim=-1)
    entropy_left = torch.log(pressure_left / density_left**gamma)
    entropy_right = torch.log(pressure_right / density_right**gamma)
    return F.relu(entropy_right - entropy_left)
