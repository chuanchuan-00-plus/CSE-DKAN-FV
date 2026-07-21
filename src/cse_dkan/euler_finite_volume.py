"""Positivity-audited HLLC finite volume backbone for one-dimensional Euler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .finite_volume import minmod


def primitive_to_conservative(primitive: Tensor, gamma: float = 1.4) -> Tensor:
    density, velocity, pressure = primitive.unbind(dim=-1)
    energy = pressure / (gamma - 1.0) + 0.5 * density * velocity**2
    return torch.stack((density, density * velocity, energy), dim=-1)


def conservative_to_primitive(conservative: Tensor, gamma: float = 1.4) -> Tensor:
    density, momentum, energy = conservative.unbind(dim=-1)
    velocity = momentum / density
    pressure = (gamma - 1.0) * (energy - 0.5 * momentum * velocity)
    return torch.stack((density, velocity, pressure), dim=-1)


def euler_flux(conservative: Tensor, gamma: float = 1.4) -> Tensor:
    primitive = conservative_to_primitive(conservative, gamma)
    density, velocity, pressure = primitive.unbind(dim=-1)
    energy = conservative[..., 2]
    return torch.stack(
        (density * velocity, density * velocity**2 + pressure, velocity * (energy + pressure)),
        dim=-1,
    )


def hllc_flux(left: Tensor, right: Tensor, gamma: float = 1.4) -> Tensor:
    """Toro HLLC flux with Davis outer-wave estimates."""
    primitive_left = conservative_to_primitive(left, gamma)
    primitive_right = conservative_to_primitive(right, gamma)
    rho_l, vel_l, pressure_l = primitive_left.unbind(dim=-1)
    rho_r, vel_r, pressure_r = primitive_right.unbind(dim=-1)
    sound_l = torch.sqrt(gamma * pressure_l / rho_l)
    sound_r = torch.sqrt(gamma * pressure_r / rho_r)
    speed_l = torch.minimum(vel_l - sound_l, vel_r - sound_r)
    speed_r = torch.maximum(vel_l + sound_l, vel_r + sound_r)
    denominator = rho_l * (speed_l - vel_l) - rho_r * (speed_r - vel_r)
    speed_m = (
        pressure_r
        - pressure_l
        + rho_l * vel_l * (speed_l - vel_l)
        - rho_r * vel_r * (speed_r - vel_r)
    ) / denominator

    def star_state(
        conservative: Tensor,
        density: Tensor,
        velocity: Tensor,
        pressure: Tensor,
        outer_speed: Tensor,
    ) -> Tensor:
        star_density = density * (outer_speed - velocity) / (outer_speed - speed_m)
        specific_energy = conservative[..., 2] / density
        star_energy = star_density * (
            specific_energy
            + (speed_m - velocity)
            * (speed_m + pressure / (density * (outer_speed - velocity)))
        )
        return torch.stack((star_density, star_density * speed_m, star_energy), dim=-1)

    flux_left = euler_flux(left, gamma)
    flux_right = euler_flux(right, gamma)
    star_left = star_state(left, rho_l, vel_l, pressure_l, speed_l)
    star_right = star_state(right, rho_r, vel_r, pressure_r, speed_r)
    flux_star_left = flux_left + speed_l.unsqueeze(-1) * (star_left - left)
    flux_star_right = flux_right + speed_r.unsqueeze(-1) * (star_right - right)
    return torch.where(
        (speed_l >= 0.0).unsqueeze(-1),
        flux_left,
        torch.where(
            (speed_m >= 0.0).unsqueeze(-1),
            flux_star_left,
            torch.where((speed_r > 0.0).unsqueeze(-1), flux_star_right, flux_right),
        ),
    )


@dataclass(frozen=True)
class EulerHLLCFV:
    x_left: float = 0.0
    x_right: float = 1.0
    n_cells: int = 400
    gamma: float = 1.4
    cfl: float = 0.35
    boundary: str = "outflow"
    reconstruction: str | nn.Module = "first_order"
    positivity_floor: float = 1.0e-10

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.n_cells

    @property
    def x(self) -> Tensor:
        return self.x_left + (torch.arange(self.n_cells) + 0.5) * self.dx

    def _primitive_slope(self, primitive: Tensor) -> Tensor:
        if self.reconstruction == "first_order":
            return torch.zeros_like(primitive)
        if self.boundary == "periodic":
            left_difference = primitive - torch.roll(primitive, 1, dims=-2)
            right_difference = torch.roll(primitive, -1, dims=-2) - primitive
        elif self.boundary in {"outflow", "reflecting"}:
            left_ghost = primitive[..., :1, :].clone()
            right_ghost = primitive[..., -1:, :].clone()
            if self.boundary == "reflecting":
                left_ghost[..., 1] = -left_ghost[..., 1]
                right_ghost[..., 1] = -right_ghost[..., 1]
            left_difference = primitive - torch.cat(
                (left_ghost, primitive[..., :-1, :]), dim=-2
            )
            right_difference = torch.cat(
                (primitive[..., 1:, :], right_ghost), dim=-2
            ) - primitive
        else:
            raise ValueError(f"unknown boundary: {self.boundary}")
        slope = minmod(
            2.0 * left_difference,
            0.5 * (left_difference + right_difference),
            2.0 * right_difference,
        )
        if isinstance(self.reconstruction, nn.Module):
            slope = self.reconstruction(left_difference, right_difference) * slope
        elif self.reconstruction != "muscl_mc":
            raise ValueError(f"unknown reconstruction: {self.reconstruction}")

        # Scale the whole primitive slope so both edge density and pressure
        # remain positive without changing the cell average.
        theta = torch.ones_like(primitive[..., :1])
        for component in (0, 2):
            value = primitive[..., component : component + 1]
            change = 0.5 * torch.abs(slope[..., component : component + 1])
            admissible = (value - self.positivity_floor) / torch.clamp(change, min=1.0e-30)
            theta = torch.minimum(theta, torch.clamp(admissible, 0.0, 1.0))
        return theta * slope

    def interface_flux(self, state: Tensor) -> Tensor:
        primitive = conservative_to_primitive(state, self.gamma)
        slope = self._primitive_slope(primitive)
        if self.boundary == "periodic":
            left_primitive = primitive + 0.5 * slope
            right_primitive = torch.roll(primitive, -1, dims=-2) - 0.5 * torch.roll(
                slope, -1, dims=-2
            )
            return hllc_flux(
                primitive_to_conservative(left_primitive, self.gamma),
                primitive_to_conservative(right_primitive, self.gamma),
                self.gamma,
            )
        if self.boundary in {"outflow", "reflecting"}:
            left_edge = primitive - 0.5 * slope
            right_edge = primitive + 0.5 * slope
            left_ghost = primitive[..., :1, :].clone()
            right_ghost = primitive[..., -1:, :].clone()
            if self.boundary == "reflecting":
                left_ghost = left_edge[..., :1, :].clone()
                right_ghost = right_edge[..., -1:, :].clone()
                left_ghost[..., 1] = -left_ghost[..., 1]
                right_ghost[..., 1] = -right_ghost[..., 1]
            left_primitive = torch.cat(
                (left_ghost, right_edge), dim=-2
            )
            right_primitive = torch.cat(
                (left_edge, right_ghost), dim=-2
            )
            return hllc_flux(
                primitive_to_conservative(left_primitive, self.gamma),
                primitive_to_conservative(right_primitive, self.gamma),
                self.gamma,
            )
        raise ValueError(f"unknown boundary: {self.boundary}")

    def rhs(self, state: Tensor) -> Tensor:
        flux = self.interface_flux(state)
        if self.boundary == "periodic":
            return -(flux - torch.roll(flux, 1, dims=-2)) / self.dx
        if self.boundary in {"outflow", "reflecting"}:
            return -(flux[..., 1:, :] - flux[..., :-1, :]) / self.dx
        raise ValueError(f"unknown boundary: {self.boundary}")

    def stable_dt(self, state: Tensor) -> float:
        primitive = conservative_to_primitive(state, self.gamma)
        density, velocity, pressure = primitive.unbind(dim=-1)
        if torch.any(density <= 0.0) or torch.any(pressure <= 0.0):
            raise FloatingPointError("nonpositive Euler state")
        speed = torch.abs(velocity) + torch.sqrt(self.gamma * pressure / density)
        return self.cfl * self.dx / max(float(torch.max(speed).detach()), 1.0e-12)

    def step(self, state: Tensor, dt: float) -> Tensor:
        first = state + dt * self.rhs(state)
        second = 0.75 * state + 0.25 * (first + dt * self.rhs(first))
        updated = (1.0 / 3.0) * state + (2.0 / 3.0) * (second + dt * self.rhs(second))
        primitive = conservative_to_primitive(updated, self.gamma)
        if torch.any(primitive[..., 0] <= 0.0) or torch.any(primitive[..., 2] <= 0.0):
            raise FloatingPointError("HLLC step lost positivity; reduce CFL or add a positivity limiter")
        return updated

    def solve(self, initial_state: Tensor, final_time: float) -> Tensor:
        if initial_state.shape[-2:] != (self.n_cells, 3):
            raise ValueError("initial_state must end in (n_cells, 3)")
        state = initial_state
        time = 0.0
        while time < final_time - 1.0e-15:
            dt = min(self.stable_dt(state), final_time - time)
            state = self.step(state, dt)
            time += dt
        return state
