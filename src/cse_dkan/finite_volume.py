"""Locally conservative Burgers backbone with an optional DKAN TVD limiter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .models import DKAN


def burgers_godunov_flux(left: Tensor, right: Tensor) -> Tensor:
    """Entropy-stable exact Godunov flux for f(u)=u^2/2."""
    flux_left = 0.5 * left**2
    flux_right = 0.5 * right**2
    rarefaction = left <= right
    rarefaction_flux = torch.where(
        left >= 0.0,
        flux_left,
        torch.where(right <= 0.0, flux_right, torch.zeros_like(left)),
    )
    shock_speed = 0.5 * (left + right)
    shock_flux = torch.where(shock_speed >= 0.0, flux_left, flux_right)
    return torch.where(rarefaction, rarefaction_flux, shock_flux)


def minmod(*values: Tensor) -> Tensor:
    if not values:
        raise ValueError("minmod requires at least one tensor")
    stacked = torch.stack(values, dim=0)
    same_positive = torch.all(stacked > 0.0, dim=0)
    same_negative = torch.all(stacked < 0.0, dim=0)
    magnitude = torch.min(torch.abs(stacked), dim=0).values
    return torch.where(
        same_positive,
        magnitude,
        torch.where(same_negative, -magnitude, torch.zeros_like(magnitude)),
    )


class DKANTVDLimiter(nn.Module):
    """Learn only a [0,1] scaling of a provably TVD MC slope."""

    def __init__(self, hidden_width: int = 8) -> None:
        super().__init__()
        self.network = DKAN([3, hidden_width, 1], grid_size=5, spline_order=3)
        self.raw_baseline = nn.Parameter(torch.tensor(2.0))

    def forward(self, left_difference: Tensor, right_difference: Tensor) -> Tensor:
        scale = torch.maximum(
            torch.maximum(torch.abs(left_difference), torch.abs(right_difference)),
            torch.full_like(left_difference, 1.0e-6),
        )
        features = torch.stack(
            (
                left_difference / scale,
                right_difference / scale,
                (right_difference - left_difference) / scale,
            ),
            dim=-1,
        )
        correction = self.network(features.reshape(-1, 3)).reshape_as(left_difference)
        return torch.sigmoid(self.raw_baseline + correction)


class DKANWENO5Reconstruction(nn.Module):
    """WENO5-Z reconstruction with bounded DKAN log-weight corrections.

    The analytical WENO-Z weights are the anchor.  DKAN sees normalized local
    smoothness information and may change each log-weight by at most
    ``correction_scale``.  Softmax keeps the weights positive and summing to
    one; a local bound projection prevents reconstructed interface states from
    leaving the five-cell stencil range.
    """

    def __init__(self, hidden_width: int = 8, correction_scale: float = 0.35) -> None:
        super().__init__()
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative")
        self.network = DKAN([5, hidden_width, 3], grid_size=5, spline_order=3)
        self.correction_scale = float(correction_scale)

    @staticmethod
    def _candidates_and_smoothness(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        fm2 = torch.roll(values, 2, dims=-1)
        fm1 = torch.roll(values, 1, dims=-1)
        f0 = values
        fp1 = torch.roll(values, -1, dims=-1)
        fp2 = torch.roll(values, -2, dims=-1)
        candidates = torch.stack(
            (
                (2.0 * fm2 - 7.0 * fm1 + 11.0 * f0) / 6.0,
                (-fm1 + 5.0 * f0 + 2.0 * fp1) / 6.0,
                (2.0 * f0 + 5.0 * fp1 - fp2) / 6.0,
            ),
            dim=-1,
        )
        beta0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) ** 2 + 0.25 * (
            fm2 - 4.0 * fm1 + 3.0 * f0
        ) ** 2
        beta1 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
        beta2 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) ** 2 + 0.25 * (
            3.0 * f0 - 4.0 * fp1 + fp2
        ) ** 2
        beta = torch.stack((beta0, beta1, beta2), dim=-1)
        stencil = torch.stack((fm2, fm1, f0, fp1, fp2), dim=-1)
        return candidates, beta, stencil

    def _left(self, values: Tensor) -> Tensor:
        candidates, beta, stencil = self._candidates_and_smoothness(values)
        tau5 = torch.abs(beta[..., :1] - beta[..., 2:3])
        eps = torch.finfo(values.dtype).eps
        linear = torch.tensor([0.1, 0.6, 0.3], dtype=values.dtype, device=values.device)
        alpha = linear * (1.0 + (tau5 / (beta + eps)) ** 2)
        analytical = alpha / torch.sum(alpha, dim=-1, keepdim=True)

        beta_scale = torch.sum(beta, dim=-1, keepdim=True) + eps
        normalized_beta = beta / beta_scale
        variation = torch.amax(stencil, dim=-1) - torch.amin(stencil, dim=-1)
        magnitude = torch.amax(torch.abs(stencil), dim=-1) + eps
        features = torch.cat(
            (
                normalized_beta,
                tau5 / beta_scale,
                (variation / magnitude).unsqueeze(-1),
            ),
            dim=-1,
        )
        correction = self.correction_scale * torch.tanh(
            self.network(features.reshape(-1, 5)).reshape_as(beta)
        )
        weights = torch.softmax(torch.log(analytical + eps) + correction, dim=-1)
        reconstructed = torch.sum(weights * candidates, dim=-1)
        lower = torch.amin(stencil, dim=-1)
        upper = torch.amax(stencil, dim=-1)
        return torch.maximum(lower, torch.minimum(upper, reconstructed))

    def forward(self, cell_average: Tensor) -> tuple[Tensor, Tensor]:
        left = self._left(cell_average)
        reversed_values = torch.flip(cell_average, dims=(-1,))
        reversed_left = self._left(reversed_values)
        right = torch.roll(torch.flip(reversed_left, dims=(-1,)), -1, dims=-1)
        return left, right


@dataclass
class ConservativeBurgersFV:
    """SSP-RK3 finite volume update with one shared flux per interface."""

    x_left: float = -1.0
    x_right: float = 1.0
    n_cells: int = 256
    cfl: float = 0.35
    limiter: nn.Module | str | None = "tvd"
    viscosity: float = 0.0

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.n_cells

    def _slope(self, cell_average: Tensor) -> Tensor:
        left_difference = cell_average - torch.roll(cell_average, 1, dims=-1)
        right_difference = torch.roll(cell_average, -1, dims=-1) - cell_average
        mc_slope = minmod(
            2.0 * left_difference,
            0.5 * (left_difference + right_difference),
            2.0 * right_difference,
        )
        if self.limiter is None or self.limiter == "first_order":
            return torch.zeros_like(cell_average)
        if self.limiter == "tvd":
            return mc_slope
        if isinstance(self.limiter, nn.Module):
            return self.limiter(left_difference, right_difference) * mc_slope
        raise ValueError(f"unknown limiter: {self.limiter}")

    def interface_flux(self, cell_average: Tensor) -> Tensor:
        if isinstance(self.limiter, DKANWENO5Reconstruction):
            left_state, right_state = self.limiter(cell_average)
        else:
            slope = self._slope(cell_average)
            left_state = cell_average + 0.5 * slope
            right_state = torch.roll(cell_average, -1, dims=-1) - 0.5 * torch.roll(
                slope, -1, dims=-1
            )
        flux = burgers_godunov_flux(left_state, right_state)
        if self.viscosity > 0.0:
            # One shared total flux per face preserves finite-volume mass
            # conservation exactly, including the central diffusive flux.
            diffusive_flux = -self.viscosity * (
                torch.roll(cell_average, -1, dims=-1) - cell_average
            ) / self.dx
            flux = flux + diffusive_flux
        return flux

    def rhs(self, cell_average: Tensor) -> Tensor:
        flux = self.interface_flux(cell_average)
        return -(flux - torch.roll(flux, 1, dims=-1)) / self.dx

    def stable_dt(self, cell_average: Tensor) -> float:
        maximum_speed = max(float(torch.max(torch.abs(cell_average)).detach()), 1.0e-12)
        advective = self.cfl * self.dx / maximum_speed
        if self.viscosity <= 0.0:
            return advective
        diffusive = 0.40 * self.dx**2 / self.viscosity
        return min(advective, diffusive)

    def step(self, cell_average: Tensor, dt: float) -> Tensor:
        first = cell_average + dt * self.rhs(cell_average)
        second = 0.75 * cell_average + 0.25 * (first + dt * self.rhs(first))
        return (1.0 / 3.0) * cell_average + (2.0 / 3.0) * (second + dt * self.rhs(second))

    def solve(self, initial_cell_average: Tensor, final_time: float) -> Tensor:
        if initial_cell_average.shape[-1] != self.n_cells:
            raise ValueError("last dimension must equal n_cells")
        state = initial_cell_average
        time = 0.0
        while time < final_time - 1.0e-15:
            dt = min(self.stable_dt(state), final_time - time)
            state = self.step(state, dt)
            time += dt
        return state
