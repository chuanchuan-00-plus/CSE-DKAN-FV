"""Hard-conservative, locally bounded DKAN subcell reconstruction."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .finite_volume import minmod
from .models import DKAN


class ConservativeDKANSubcell(nn.Module):
    """Reconstruct subcell values without changing any cell average.

    A TVD-MC linear profile is augmented by a DKAN residual.  Both pieces have
    zero discrete mean on the symmetric subcell points.  A single per-cell
    scaling then keeps every reconstructed value inside the five-cell stencil
    range, preserving zero mean and therefore exact finite-volume conservation.
    """

    def __init__(self, hidden_width: int = 16, correction_scale: float = 0.35) -> None:
        super().__init__()
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative")
        self.network = DKAN([8, hidden_width, 1], grid_size=5, spline_order=3)
        self.correction_scale = float(correction_scale)

    @staticmethod
    def uniform_coordinates(refinement: int, *, dtype=None, device=None) -> Tensor:
        if refinement < 2:
            raise ValueError("refinement must be at least two")
        return (
            (torch.arange(refinement, dtype=dtype, device=device) + 0.5) / refinement
            - 0.5
        )

    def forward(
        self,
        cell_average: Tensor,
        subcell_coordinates: Tensor,
        *,
        viscosity: float,
        dx: float,
        shock_sensor_threshold: float | None = None,
        shock_sensor_minimum_jump: float = 0.0,
        shock_sensor_dilation: int = 1,
        learned_correction_multiplier: float = 1.0,
        maximum_viscous_transition_cells: float | None = None,
    ) -> Tensor:
        if cell_average.ndim < 1:
            raise ValueError("cell_average must have a cell dimension")
        if subcell_coordinates.ndim != 1:
            raise ValueError("subcell_coordinates must be one-dimensional")
        if not 0.0 <= learned_correction_multiplier <= 1.0:
            raise ValueError("learned_correction_multiplier must lie in [0, 1]")
        if (
            maximum_viscous_transition_cells is not None
            and maximum_viscous_transition_cells < 0.0
        ):
            raise ValueError("maximum_viscous_transition_cells must be non-negative")
        if not torch.allclose(
            torch.mean(subcell_coordinates),
            torch.zeros((), dtype=subcell_coordinates.dtype, device=subcell_coordinates.device),
            atol=10.0 * torch.finfo(subcell_coordinates.dtype).eps,
            rtol=0.0,
        ):
            raise ValueError("subcell coordinates must have zero mean")

        center = cell_average
        fm2 = torch.roll(center, 2, dims=-1)
        fm1 = torch.roll(center, 1, dims=-1)
        fp1 = torch.roll(center, -1, dims=-1)
        fp2 = torch.roll(center, -2, dims=-1)
        differences = torch.stack((fm2 - center, fm1 - center, fp1 - center, fp2 - center), dim=-1)
        scale = torch.clamp(torch.amax(torch.abs(differences), dim=-1), min=1.0e-5)
        magnitude = torch.abs(center) + scale + 1.0e-5
        left_difference = center - fm1
        right_difference = fp1 - center
        slope = minmod(
            2.0 * left_difference,
            0.5 * (left_difference + right_difference),
            2.0 * right_difference,
        )

        q = subcell_coordinates.numel()
        xi = subcell_coordinates.reshape(*([1] * center.ndim), q)
        expanded_difference = differences.unsqueeze(-2).expand(*center.shape, q, 4)
        normalized_difference = expanded_difference / scale.unsqueeze(-1).unsqueeze(-1)
        center_feature = (center / magnitude).unsqueeze(-1).unsqueeze(-1).expand(*center.shape, q, 1)
        scale_feature = (scale / magnitude).unsqueeze(-1).unsqueeze(-1).expand(*center.shape, q, 1)
        xi_feature = (2.0 * xi).expand(*center.shape, q).unsqueeze(-1)
        log_viscosity = math.log10(max(float(viscosity) / float(dx), 1.0e-8))
        viscosity_feature = torch.full_like(xi_feature, math.tanh(0.25 * log_viscosity))
        features = torch.cat(
            (
                normalized_difference,
                center_feature,
                scale_feature,
                xi_feature,
                viscosity_feature,
            ),
            dim=-1,
        )
        raw = torch.tanh(self.network(features.reshape(-1, 8))).reshape(*center.shape, q)
        zero_mean_raw = raw - torch.mean(raw, dim=-1, keepdim=True)
        learned_correction = (
            learned_correction_multiplier
            * self.correction_scale
            * scale.unsqueeze(-1)
            * zero_mean_raw
        )
        if shock_sensor_threshold is not None:
            sensor = self.jump_sensor(
                center,
                threshold=shock_sensor_threshold,
                minimum_jump=shock_sensor_minimum_jump,
                dilation=shock_sensor_dilation,
            )
            if maximum_viscous_transition_cells is not None and viscosity > 0.0:
                dynamic_range = torch.amax(center, dim=-1, keepdim=True) - torch.amin(
                    center, dim=-1, keepdim=True
                )
                # A viscous Burgers travelling wave has a 10--90% thickness
                # 8*atanh(0.8)*nu/Delta-u.  Once that thickness is already
                # resolved by several cells, the discontinuous expert is
                # physically inappropriate and the fixed MC expert is safer.
                transition_cells = (
                    8.0
                    * math.atanh(0.8)
                    * float(viscosity)
                    / (torch.clamp(dynamic_range, min=1.0e-12) * float(dx))
                )
                sensor = sensor * (
                    transition_cells <= maximum_viscous_transition_cells
                ).to(dtype=center.dtype)
            learned_correction = learned_correction * sensor.unsqueeze(-1)
        correction = slope.unsqueeze(-1) * xi + learned_correction

        stencil = torch.stack((fm2, fm1, center, fp1, fp2), dim=-1)
        lower = torch.amin(stencil, dim=-1)
        upper = torch.amax(stencil, dim=-1)
        maximum_positive = torch.clamp(torch.amax(correction, dim=-1), min=1.0e-12)
        maximum_negative = torch.clamp(-torch.amin(correction, dim=-1), min=1.0e-12)
        theta = torch.minimum(
            torch.ones_like(center),
            torch.minimum(
                (upper - center) / maximum_positive,
                (center - lower) / maximum_negative,
            ),
        )
        theta = torch.clamp(theta, 0.0, 1.0)
        return center.unsqueeze(-1) + theta.unsqueeze(-1) * correction

    @staticmethod
    def jump_sensor(
        cell_average: Tensor,
        *,
        threshold: float = 0.3,
        minimum_jump: float = 0.0,
        dilation: int = 1,
    ) -> Tensor:
        """Flag cells touching a large normalized jump.

        The sensor is dimensionless and amplitude invariant.  It is used only
        as an inference safety gate: inactive cells exactly recover the fixed
        TVD-MC profile, while active cells retain the learned zero-mean DKAN
        correction.  Dilation includes immediate shock-neighbour cells.
        """
        if threshold < 0.0:
            raise ValueError("shock sensor threshold must be non-negative")
        if minimum_jump < 0.0:
            raise ValueError("minimum jump must be non-negative")
        if dilation < 0:
            raise ValueError("shock sensor dilation must be non-negative")
        dynamic_range = torch.amax(cell_average, dim=-1, keepdim=True) - torch.amin(
            cell_average, dim=-1, keepdim=True
        )
        interface_jump = torch.abs(
            cell_average - torch.roll(cell_average, 1, dims=-1)
        ) / torch.clamp(dynamic_range, min=1.0e-12)
        strength = torch.maximum(interface_jump, torch.roll(interface_jump, -1, dims=-1))
        active = (strength >= threshold) & (
            torch.maximum(interface_jump, torch.roll(interface_jump, -1, dims=-1))
            * torch.clamp(dynamic_range, min=1.0e-12)
            >= minimum_jump
        )
        for _ in range(dilation):
            active = active | torch.roll(active, 1, dims=-1) | torch.roll(
                active, -1, dims=-1
            )
        return active.to(dtype=cell_average.dtype)

    @staticmethod
    def project_total_variation(
        reconstructed: Tensor,
        cell_average: Tensor,
        relative_budget: float = 0.01,
        absolute_budget: Tensor | float | None = None,
    ) -> Tensor:
        """Apply a conservative convex safety projection to a global TV budget."""
        if relative_budget < 0.0:
            raise ValueError("relative TV budget must be non-negative")
        if reconstructed.shape[:-1] != cell_average.shape:
            raise ValueError("reconstructed shape must be cell_average shape plus subcells")
        base = cell_average.unsqueeze(-1).expand_as(reconstructed)
        flat_reconstructed = reconstructed.reshape(*cell_average.shape[:-1], -1)
        flat_base = base.reshape(*cell_average.shape[:-1], -1)
        reconstructed_tv = torch.sum(
            torch.abs(flat_reconstructed - torch.roll(flat_reconstructed, 1, dims=-1)),
            dim=-1,
        )
        base_tv = torch.sum(
            torch.abs(flat_base - torch.roll(flat_base, 1, dims=-1)), dim=-1
        )
        if absolute_budget is None:
            budget = (1.0 + relative_budget) * base_tv
        else:
            budget = torch.as_tensor(
                absolute_budget, dtype=base_tv.dtype, device=base_tv.device
            )
            budget = torch.broadcast_to(budget, base_tv.shape)
            if torch.any(budget < base_tv):
                raise ValueError("absolute TV budget cannot be below piecewise-constant TV")
        denominator = torch.clamp(reconstructed_tv - base_tv, min=1.0e-12)
        admissible = torch.clamp((budget - base_tv) / denominator, 0.0, 1.0)
        alpha = torch.where(reconstructed_tv <= budget, torch.ones_like(admissible), admissible)
        return base + alpha.reshape(*alpha.shape, 1, 1) * (reconstructed - base)
