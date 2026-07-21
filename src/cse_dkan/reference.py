"""Deterministic one-dimensional reference solvers.

The WENO implementation is intentionally self-contained.  It provides a
conservative high-resolution reference for Burgers experiments and is covered
by constant-state and conservation tests.  The Sod solver follows the exact
Riemann construction for an ideal gas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


def _weno5z_left(values: Array, eps: float = 1.0e-40, power: int = 2) -> Array:
    """Reconstruct a periodic left state at every i+1/2 interface."""
    fm2 = np.roll(values, 2)
    fm1 = np.roll(values, 1)
    f0 = values
    fp1 = np.roll(values, -1)
    fp2 = np.roll(values, -2)

    p0 = (2.0 * fm2 - 7.0 * fm1 + 11.0 * f0) / 6.0
    p1 = (-fm1 + 5.0 * f0 + 2.0 * fp1) / 6.0
    p2 = (2.0 * f0 + 5.0 * fp1 - fp2) / 6.0

    b0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) ** 2 + 0.25 * (
        fm2 - 4.0 * fm1 + 3.0 * f0
    ) ** 2
    b1 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
    b2 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) ** 2 + 0.25 * (
        3.0 * f0 - 4.0 * fp1 + fp2
    ) ** 2

    tau5 = np.abs(b0 - b2)
    alpha0 = 0.1 * (1.0 + (tau5 / (b0 + eps)) ** power)
    alpha1 = 0.6 * (1.0 + (tau5 / (b1 + eps)) ** power)
    alpha2 = 0.3 * (1.0 + (tau5 / (b2 + eps)) ** power)
    alpha_sum = alpha0 + alpha1 + alpha2
    return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / alpha_sum


def _weno5z_right(values: Array) -> Array:
    """Reconstruct a periodic right state at every i+1/2 interface."""
    reversed_left = _weno5z_left(values[::-1])
    return np.roll(reversed_left[::-1], -1)


def burgers_weno_rhs(u: Array, dx: float, viscosity: float = 0.0) -> Array:
    """Semi-discrete WENO5-Z right-hand side for Burgers' equation."""
    flux = 0.5 * u**2
    alpha = max(float(np.max(np.abs(u))), 1.0e-14)
    flux_plus = 0.5 * (flux + alpha * u)
    flux_minus = 0.5 * (flux - alpha * u)
    numerical_flux = _weno5z_left(flux_plus) + _weno5z_right(flux_minus)
    rhs = -(numerical_flux - np.roll(numerical_flux, 1)) / dx
    if viscosity > 0.0:
        rhs = rhs + viscosity * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
    return rhs


@dataclass(frozen=True)
class BurgersWENO:
    """Periodic WENO5-Z + SSP-RK3 reference solver for Burgers' equation."""

    x_left: float = 0.0
    x_right: float = 2.0
    n_cells: int = 1024
    cfl: float = 0.35
    viscosity: float = 0.0

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.n_cells

    @property
    def x(self) -> Array:
        return self.x_left + (np.arange(self.n_cells) + 0.5) * self.dx

    def _stable_dt(self, u: Array) -> float:
        advective = self.cfl * self.dx / max(float(np.max(np.abs(u))), 1.0e-12)
        if self.viscosity <= 0.0:
            return advective
        diffusive = 0.40 * self.dx**2 / self.viscosity
        return min(advective, diffusive)

    def solve(
        self,
        initial_condition: Callable[[Array], Array],
        final_time: float,
        snapshot_times: tuple[float, ...] = (),
    ) -> dict[float, Array]:
        """Advance to final_time and return requested snapshots plus the final state."""
        if final_time < 0.0:
            raise ValueError("final_time must be non-negative")
        requested = sorted({float(t) for t in snapshot_times if 0.0 <= t <= final_time} | {final_time})
        u = np.asarray(initial_condition(self.x), dtype=np.float64).copy()
        if u.shape != self.x.shape:
            raise ValueError(f"initial condition returned {u.shape}, expected {self.x.shape}")
        result: dict[float, Array] = {}
        time = 0.0
        request_index = 0
        if requested and requested[0] == 0.0:
            result[0.0] = u.copy()
            request_index = 1

        while time < final_time - 1.0e-15:
            target = requested[request_index] if request_index < len(requested) else final_time
            dt = min(self._stable_dt(u), target - time, final_time - time)
            if dt <= 0.0:
                result[target] = u.copy()
                request_index += 1
                continue
            rhs = lambda state: burgers_weno_rhs(state, self.dx, self.viscosity)
            u1 = u + dt * rhs(u)
            u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
            u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))
            time += dt
            if request_index < len(requested) and abs(time - target) <= 5.0e-13:
                result[target] = u.copy()
                request_index += 1
        return result


@dataclass(frozen=True)
class PrimitiveState:
    density: float
    velocity: float
    pressure: float


class ExactSodSolver:
    """Exact ideal-gas Riemann solver, suitable for Sod and Lax shock tubes."""

    def __init__(
        self,
        left: PrimitiveState = PrimitiveState(1.0, 0.0, 1.0),
        right: PrimitiveState = PrimitiveState(0.125, 0.0, 0.1),
        gamma: float = 1.4,
        discontinuity: float = 0.5,
    ) -> None:
        if min(left.density, left.pressure, right.density, right.pressure) <= 0.0:
            raise ValueError("density and pressure must be positive")
        self.left = left
        self.right = right
        self.gamma = float(gamma)
        self.discontinuity = float(discontinuity)
        self.p_star, self.u_star = self._star_state()

    def _sound_speed(self, state: PrimitiveState) -> float:
        return float(np.sqrt(self.gamma * state.pressure / state.density))

    def _pressure_function(self, p: float, state: PrimitiveState) -> tuple[float, float]:
        g = self.gamma
        a = self._sound_speed(state)
        if p > state.pressure:
            coef_a = 2.0 / ((g + 1.0) * state.density)
            coef_b = (g - 1.0) / (g + 1.0) * state.pressure
            root = np.sqrt(coef_a / (p + coef_b))
            value = (p - state.pressure) * root
            derivative = root * (1.0 - 0.5 * (p - state.pressure) / (p + coef_b))
            return float(value), float(derivative)
        exponent = (g - 1.0) / (2.0 * g)
        ratio = p / state.pressure
        value = 2.0 * a / (g - 1.0) * (ratio**exponent - 1.0)
        derivative = (1.0 / (state.density * a)) * ratio ** (-(g + 1.0) / (2.0 * g))
        return float(value), float(derivative)

    def _star_state(self) -> tuple[float, float]:
        l, r = self.left, self.right
        a_l, a_r = self._sound_speed(l), self._sound_speed(r)
        p = max(
            1.0e-10,
            0.5 * (l.pressure + r.pressure)
            - 0.125 * (r.velocity - l.velocity) * (l.density + r.density) * (a_l + a_r),
        )
        for _ in range(64):
            f_l, d_l = self._pressure_function(p, l)
            f_r, d_r = self._pressure_function(p, r)
            p_new = max(1.0e-12, p - (f_l + f_r + r.velocity - l.velocity) / (d_l + d_r))
            if abs(p_new - p) <= 1.0e-12 * (p_new + p + 1.0):
                p = p_new
                break
            p = p_new
        f_l, _ = self._pressure_function(p, l)
        f_r, _ = self._pressure_function(p, r)
        u = 0.5 * (l.velocity + r.velocity + f_r - f_l)
        return float(p), float(u)

    def _star_density(self, state: PrimitiveState) -> float:
        ratio = self.p_star / state.pressure
        g_ratio = (self.gamma - 1.0) / (self.gamma + 1.0)
        if self.p_star > state.pressure:
            return state.density * (ratio + g_ratio) / (g_ratio * ratio + 1.0)
        return state.density * ratio ** (1.0 / self.gamma)

    @property
    def left_star_state(self) -> PrimitiveState:
        return PrimitiveState(self._star_density(self.left), self.u_star, self.p_star)

    @property
    def right_star_state(self) -> PrimitiveState:
        return PrimitiveState(self._star_density(self.right), self.u_star, self.p_star)

    def sample(self, x: Array, time: float) -> tuple[Array, Array, Array]:
        x = np.asarray(x, dtype=np.float64)
        if time <= 0.0:
            left_mask = x < self.discontinuity
            rho = np.where(left_mask, self.left.density, self.right.density)
            vel = np.where(left_mask, self.left.velocity, self.right.velocity)
            pressure = np.where(left_mask, self.left.pressure, self.right.pressure)
            return rho, vel, pressure

        xi = (x - self.discontinuity) / time
        rho = np.empty_like(xi)
        vel = np.empty_like(xi)
        pressure = np.empty_like(xi)
        for index, speed in np.ndenumerate(xi):
            state = self._sample_similarity(float(speed))
            rho[index], vel[index], pressure[index] = state
        return rho, vel, pressure

    def _sample_similarity(self, speed: float) -> tuple[float, float, float]:
        if speed <= self.u_star:
            return self._sample_left(speed)
        return self._sample_right(speed)

    def _sample_left(self, speed: float) -> tuple[float, float, float]:
        state, g = self.left, self.gamma
        a = self._sound_speed(state)
        if self.p_star > state.pressure:
            shock_speed = state.velocity - a * np.sqrt(
                (g + 1.0) / (2.0 * g) * self.p_star / state.pressure + (g - 1.0) / (2.0 * g)
            )
            if speed <= shock_speed:
                return state.density, state.velocity, state.pressure
            return self._star_density(state), self.u_star, self.p_star

        head = state.velocity - a
        a_star = a * (self.p_star / state.pressure) ** ((g - 1.0) / (2.0 * g))
        tail = self.u_star - a_star
        if speed <= head:
            return state.density, state.velocity, state.pressure
        if speed >= tail:
            return self._star_density(state), self.u_star, self.p_star
        velocity = 2.0 / (g + 1.0) * (a + 0.5 * (g - 1.0) * state.velocity + speed)
        sound = 2.0 / (g + 1.0) * (a + 0.5 * (g - 1.0) * (state.velocity - speed))
        density = state.density * (sound / a) ** (2.0 / (g - 1.0))
        pressure = state.pressure * (sound / a) ** (2.0 * g / (g - 1.0))
        return float(density), float(velocity), float(pressure)

    def _sample_right(self, speed: float) -> tuple[float, float, float]:
        state, g = self.right, self.gamma
        a = self._sound_speed(state)
        if self.p_star > state.pressure:
            shock_speed = state.velocity + a * np.sqrt(
                (g + 1.0) / (2.0 * g) * self.p_star / state.pressure + (g - 1.0) / (2.0 * g)
            )
            if speed >= shock_speed:
                return state.density, state.velocity, state.pressure
            return self._star_density(state), self.u_star, self.p_star

        head = state.velocity + a
        a_star = a * (self.p_star / state.pressure) ** ((g - 1.0) / (2.0 * g))
        tail = self.u_star + a_star
        if speed >= head:
            return state.density, state.velocity, state.pressure
        if speed <= tail:
            return self._star_density(state), self.u_star, self.p_star
        velocity = 2.0 / (g + 1.0) * (-a + 0.5 * (g - 1.0) * state.velocity + speed)
        sound = 2.0 / (g + 1.0) * (a - 0.5 * (g - 1.0) * (state.velocity - speed))
        density = state.density * (sound / a) ** (2.0 / (g - 1.0))
        pressure = state.pressure * (sound / a) ** (2.0 * g / (g - 1.0))
        return float(density), float(velocity), float(pressure)
