"""Strong, control-volume, entropy, global-conservation, and RH losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gradient(outputs: Tensor, inputs: Tensor, create_graph: bool = True) -> Tensor:
    # Constant analytical fields are valid inputs to the loss operators.
    # Keep their zero derivative connected to ``inputs`` so that a later
    # derivative (for example u_xx) is also well defined.
    if not outputs.requires_grad:
        return inputs * 0.0
    derivative = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return inputs * 0.0 if derivative is None else derivative


def burgers_strong_residual(model: nn.Module, points: Tensor, viscosity: float = 0.0) -> Tensor:
    points = points.requires_grad_(True)
    u = model(points)
    derivatives = gradient(u, points)
    u_x, u_t = derivatives[:, :1], derivatives[:, 1:2]
    residual = u_t + u * u_x
    if viscosity > 0.0:
        u_xx = gradient(u_x, points)[:, :1]
        residual = residual - viscosity * u_xx
    return residual


@dataclass(frozen=True)
class RectangularControlVolumes:
    centers: Tensor
    half_width_x: Tensor
    half_width_t: Tensor

    def validate(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("centers must have shape (n, 2)")
        n = self.centers.shape[0]
        if self.half_width_x.reshape(-1).shape[0] != n or self.half_width_t.reshape(-1).shape[0] != n:
            raise ValueError("control-volume widths must match center count")
        if torch.any(self.half_width_x <= 0.0) or torch.any(self.half_width_t <= 0.0):
            raise ValueError("control-volume half-widths must be positive")


def _gauss_legendre(order: int, like: Tensor) -> tuple[Tensor, Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        torch.as_tensor(nodes, dtype=like.dtype, device=like.device),
        torch.as_tensor(weights, dtype=like.dtype, device=like.device),
    )


def _boundary_samples(volumes: RectangularControlVolumes, quadrature_order: int) -> dict[str, tuple[Tensor, Tensor]]:
    volumes.validate()
    centers = volumes.centers
    hx = volumes.half_width_x.reshape(-1, 1)
    ht = volumes.half_width_t.reshape(-1, 1)
    nodes, weights = _gauss_legendre(quadrature_order, centers)
    q = nodes.reshape(1, -1)
    w = weights.reshape(1, -1)
    x_center, t_center = centers[:, :1], centers[:, 1:2]

    x_horizontal = x_center + hx * q
    top = torch.stack((x_horizontal, t_center + ht + torch.zeros_like(x_horizontal)), dim=-1)
    bottom = torch.stack((x_horizontal, t_center - ht + torch.zeros_like(x_horizontal)), dim=-1)
    t_vertical = t_center + ht * q
    right = torch.stack((x_center + hx + torch.zeros_like(t_vertical), t_vertical), dim=-1)
    left = torch.stack((x_center - hx + torch.zeros_like(t_vertical), t_vertical), dim=-1)
    return {
        "top": (top.reshape(-1, 2), w * hx),
        "bottom": (bottom.reshape(-1, 2), w * hx),
        "right": (right.reshape(-1, 2), w * ht),
        "left": (left.reshape(-1, 2), w * ht),
    }


def _integrate_boundary_values(values: Tensor, scaled_weights: Tensor, n_volumes: int) -> Tensor:
    components = values.shape[-1]
    q_order = values.shape[0] // n_volumes
    reshaped = values.reshape(n_volumes, q_order, components)
    return torch.sum(reshaped * scaled_weights.reshape(n_volumes, q_order, 1), dim=1)


def burgers_control_volume_balance(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    quadrature_order: int = 4,
    viscosity: float = 0.0,
) -> Tensor:
    """Return normalized conservation balance for each rectangular space-time cell."""
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]

    def state(points: Tensor) -> Tensor:
        return model(points)

    top = _integrate_boundary_values(state(samples["top"][0]), samples["top"][1], n)
    bottom = _integrate_boundary_values(state(samples["bottom"][0]), samples["bottom"][1], n)

    def flux(points: Tensor) -> Tensor:
        if viscosity > 0.0:
            with torch.enable_grad():
                points = points.requires_grad_(True)
                u = model(points)
                u_x = gradient(u, points)[:, :1]
                return 0.5 * u**2 - viscosity * u_x
        u = model(points)
        return 0.5 * u**2

    right = _integrate_boundary_values(flux(samples["right"][0]), samples["right"][1], n)
    left = _integrate_boundary_values(flux(samples["left"][0]), samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    return (top - bottom + right - left) / area


def burgers_entropy_violation(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    quadrature_order: int = 4,
) -> Tensor:
    """Positive part of the inviscid quadratic-entropy balance."""
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]

    def entropy(points: Tensor) -> Tensor:
        u = model(points)
        return 0.5 * u**2

    def entropy_flux(points: Tensor) -> Tensor:
        u = model(points)
        return u**3 / 3.0

    top = _integrate_boundary_values(entropy(samples["top"][0]), samples["top"][1], n)
    bottom = _integrate_boundary_values(entropy(samples["bottom"][0]), samples["bottom"][1], n)
    right = _integrate_boundary_values(entropy_flux(samples["right"][0]), samples["right"][1], n)
    left = _integrate_boundary_values(entropy_flux(samples["left"][0]), samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    return F.relu((top - bottom + right - left) / area)


def burgers_viscous_entropy_violation(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    viscosity: float,
    quadrature_order: int = 4,
) -> Tensor:
    """Positive viscous entropy balance using q_nu=u^3/3-nu*u*u_x."""
    if viscosity <= 0.0:
        return burgers_entropy_violation(model, volumes, quadrature_order)
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]

    def entropy(points: Tensor) -> Tensor:
        value = model(points)
        return 0.5 * value**2

    def entropy_flux(points: Tensor) -> Tensor:
        with torch.enable_grad():
            points = points.requires_grad_(True)
            value = model(points)
            derivative = gradient(value, points)[:, :1]
            return value**3 / 3.0 - viscosity * value * derivative

    top = _integrate_boundary_values(entropy(samples["top"][0]), samples["top"][1], n)
    bottom = _integrate_boundary_values(entropy(samples["bottom"][0]), samples["bottom"][1], n)
    right = _integrate_boundary_values(entropy_flux(samples["right"][0]), samples["right"][1], n)
    left = _integrate_boundary_values(entropy_flux(samples["left"][0]), samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    return F.relu((top - bottom + right - left) / area)


def burgers_control_volume_and_entropy(
    model: nn.Module,
    volumes: RectangularControlVolumes,
    quadrature_order: int = 4,
    normalization: str = "area",
) -> tuple[Tensor, Tensor]:
    """Fused inviscid conservation/entropy balances using one model call."""
    samples = _boundary_samples(volumes, quadrature_order)
    n = volumes.centers.shape[0]
    names = ("top", "bottom", "right", "left")
    counts = [samples[name][0].shape[0] for name in names]
    all_points = torch.cat([samples[name][0] for name in names], dim=0)
    states = torch.split(model(all_points), counts, dim=0)
    integrated_state = {
        name: _integrate_boundary_values(state, samples[name][1], n)
        for name, state in zip(names, states)
    }
    top, bottom = integrated_state["top"], integrated_state["bottom"]
    right_u, left_u = states[2], states[3]
    right_flux = _integrate_boundary_values(0.5 * right_u**2, samples["right"][1], n)
    left_flux = _integrate_boundary_values(0.5 * left_u**2, samples["left"][1], n)
    area = 4.0 * volumes.half_width_x.reshape(-1, 1) * volumes.half_width_t.reshape(-1, 1)
    if normalization == "area":
        normalizer = area
    elif normalization == "sqrt_area":
        normalizer = torch.sqrt(area)
    elif normalization == "integral":
        normalizer = torch.ones_like(area)
    else:
        raise ValueError(f"unknown control-volume normalization: {normalization}")
    conservation = (top - bottom + right_flux - left_flux) / normalizer

    integrated_entropy = {
        "top": _integrate_boundary_values(0.5 * states[0] ** 2, samples["top"][1], n),
        "bottom": _integrate_boundary_values(0.5 * states[1] ** 2, samples["bottom"][1], n),
        "right": _integrate_boundary_values(states[2] ** 3 / 3.0, samples["right"][1], n),
        "left": _integrate_boundary_values(states[3] ** 3 / 3.0, samples["left"][1], n),
    }
    entropy_balance = (
        integrated_entropy["top"]
        - integrated_entropy["bottom"]
        + integrated_entropy["right"]
        - integrated_entropy["left"]
    ) / normalizer
    return conservation, F.relu(entropy_balance)


def burgers_periodic_global_conservation(
    model: nn.Module,
    x: Tensor,
    times: Tensor,
    initial_condition,
) -> Tensor:
    """Mean-state drift for a periodic Burgers domain on a fixed x quadrature."""
    x = x.reshape(1, -1, 1)
    times = times.reshape(-1, 1, 1)
    x_grid = x.expand(times.shape[0], -1, -1)
    t_grid = times.expand(-1, x.shape[1], -1)
    points = torch.cat((x_grid, t_grid), dim=-1).reshape(-1, 2)
    predicted_mean = model(points).reshape(times.shape[0], x.shape[1], -1).mean(dim=1)
    initial_mean = initial_condition(x.reshape(-1, 1)).mean(dim=0, keepdim=True)
    return predicted_mean - initial_mean


def burgers_rankine_hugoniot_residual(
    model: nn.Module,
    shock_position,
    times: Tensor,
    offset: float,
) -> Tensor:
    """RH residual [f] - s[u] sampled on both sides of a learned 1D shock."""
    times = times.reshape(-1, 1).requires_grad_(True)
    position = shock_position(times)
    speed = gradient(position, times)
    left_points = torch.cat((position - offset, times), dim=-1)
    right_points = torch.cat((position + offset, times), dim=-1)
    u_left, u_right = model(left_points), model(right_points)
    jump_u = u_right - u_left
    jump_flux = 0.5 * (u_right**2 - u_left**2)
    return jump_flux - speed * jump_u


def burgers_shock_entropy_violation(
    model: nn.Module,
    shock_position,
    times: Tensor,
    offset: float,
) -> Tensor:
    """Lax/compressivity violations [u_R-u_L, s-u_L, u_R-s]_+."""
    times = times.reshape(-1, 1).requires_grad_(True)
    position = shock_position(times)
    speed = gradient(position, times)
    left_points = torch.cat((position - offset, times), dim=-1)
    right_points = torch.cat((position + offset, times), dim=-1)
    u_left, u_right = model(left_points), model(right_points)
    return torch.cat(
        (
            F.relu(u_right - u_left),
            F.relu(speed - u_left),
            F.relu(u_right - speed),
        ),
        dim=-1,
    )
