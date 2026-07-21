"""Hard-conservative and positivity-preserving DKAN subcells for 1D Euler."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .euler_finite_volume import conservative_to_primitive, primitive_to_conservative
from .finite_volume import minmod
from .models import DKAN


def conservative_density_total_variation(profile: Tensor) -> Tensor:
    """Density TV along an ordered cell-by-subcell conservative profile."""
    if profile.ndim < 3 or profile.shape[-1] != 3:
        raise ValueError("profile must end in (cells, subcells, 3)")
    density = profile[..., 0].flatten(start_dim=-2)
    return torch.sum(torch.abs(density[..., 1:] - density[..., :-1]), dim=-1)


def density_tv_trust_projection(
    fixed: Tensor,
    candidate: Tensor,
    *,
    relative_budget: float = 0.02,
    absolute_tolerance: float = 1.0e-12,
    iterations: int = 32,
) -> tuple[Tensor, Tensor]:
    """Project a candidate toward a trusted profile until density TV is admissible.

    The projection uses one maximum admissible convex multiplier per leading
    batch item.  Convex blending retains the shared parent-cell means and the
    convex admissible set of positive-density/positive-pressure Euler states.
    """
    if fixed.shape != candidate.shape:
        raise ValueError("fixed and candidate profiles must have identical shapes")
    if not 0.0 <= relative_budget:
        raise ValueError("relative_budget must be non-negative")
    if absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    fixed_tv = conservative_density_total_variation(fixed)
    candidate_tv = conservative_density_total_variation(candidate)
    budget = (1.0 + relative_budget) * fixed_tv + absolute_tolerance
    active = candidate_tv > budget
    low = torch.zeros_like(fixed_tv)
    high = torch.ones_like(fixed_tv)
    difference = candidate - fixed
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        profile = fixed + middle[..., None, None, None] * difference
        admissible = conservative_density_total_variation(profile) <= budget
        low = torch.where(active & admissible, middle, low)
        high = torch.where(active & ~admissible, middle, high)
    multiplier = torch.where(active, low, torch.ones_like(low))
    trusted = fixed + multiplier[..., None, None, None] * difference
    return trusted, multiplier


def _outflow_shift(values: Tensor, offset: int) -> Tensor:
    """Shift along the last axis and repeat the boundary value."""
    if offset == 0:
        return values
    if offset > 0:
        return torch.cat(
            (
                values[..., :1].expand(*values.shape[:-1], offset),
                values[..., :-offset],
            ),
            dim=-1,
        )
    count = -offset
    return torch.cat(
        (
            values[..., count:],
            values[..., -1:].expand(*values.shape[:-1], count),
        ),
        dim=-1,
    )


def euler_discontinuity_sensor(
    cell_average: Tensor,
    *,
    gamma: float = 1.4,
    positivity_floor: float = 1.0e-8,
    pressure_low: float = 0.025,
    pressure_high: float = 0.15,
    compression_low: float = 0.02,
    compression_high: float = 0.15,
    contact_low: float = 0.15,
    contact_high: float = 0.35,
) -> Tensor:
    """Return a bounded shock/contact activation from local primitive jumps.

    Pressure jumps and compressive velocity differences detect shocks.  A
    deliberately higher density-only threshold admits isolated contacts but
    rejects resolved entropy waves.  The piecewise-linear ramps make every
    threshold explicit and keep the sensor independent of learned weights.
    """
    if cell_average.shape[-1] != 3:
        raise ValueError("cell_average must end in three conservative components")
    primitive = conservative_to_primitive(cell_average, gamma)
    density, velocity, pressure = primitive.unbind(dim=-1)
    density_left = _outflow_shift(density, 1)
    density_right = _outflow_shift(density, -1)
    velocity_left = _outflow_shift(velocity, 1)
    velocity_right = _outflow_shift(velocity, -1)
    pressure_left = _outflow_shift(pressure, 1)
    pressure_right = _outflow_shift(pressure, -1)

    density_scale = torch.clamp(
        torch.maximum(density, torch.maximum(density_left, density_right)),
        min=positivity_floor,
    )
    pressure_scale = torch.clamp(
        torch.maximum(pressure, torch.maximum(pressure_left, pressure_right)),
        min=positivity_floor,
    )
    density_jump = torch.maximum(
        torch.abs(density - density_left), torch.abs(density_right - density)
    ) / density_scale
    pressure_jump = torch.maximum(
        torch.abs(pressure - pressure_left), torch.abs(pressure_right - pressure)
    ) / pressure_scale
    sound = torch.sqrt(
        gamma * torch.clamp(pressure, min=positivity_floor)
        / torch.clamp(density, min=positivity_floor)
    )
    compression = torch.relu(velocity_left - velocity_right) / torch.clamp(
        sound + torch.abs(velocity), min=1.0e-8
    )

    def ramp(value: Tensor, low: float, high: float) -> Tensor:
        if not 0.0 <= low < high:
            raise ValueError("sensor thresholds must satisfy 0 <= low < high")
        return torch.clamp((value - low) / (high - low), 0.0, 1.0)

    shock = torch.maximum(
        ramp(pressure_jump, pressure_low, pressure_high),
        ramp(compression, compression_low, compression_high),
    )
    contact = ramp(density_jump, contact_low, contact_high)
    return torch.maximum(shock, contact)


class ConservativeEulerDKANSubcell(nn.Module):
    """Reconstruct conservative Euler states with exact cell means.

    Three component-specific DKANs augment a conservative-variable TVD-MC
    profile.  A shared per-cell convex limiter enforces five-cell component
    bounds and positive density/pressure without changing any cell average.
    """

    def __init__(
        self,
        hidden_width: int = 16,
        correction_scale: float = 0.35,
        gamma: float = 1.4,
        positivity_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative")
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than one")
        if positivity_floor <= 0.0:
            raise ValueError("positivity_floor must be positive")
        self.networks = nn.ModuleList(
            [DKAN([10, hidden_width, 1], grid_size=5, spline_order=3) for _ in range(3)]
        )
        # Begin as the trusted TVD-MC baseline.  The residual is admitted only
        # after training supplies evidence for a nonzero correction.
        for network in self.networks:
            output_layer = network.layers[-1]
            nn.init.zeros_(output_layer.spline_coefficients)
            nn.init.zeros_(output_layer.dyt_weight)
            nn.init.zeros_(output_layer.bias)
        self.correction_scale = float(correction_scale)
        self.gamma = float(gamma)
        self.positivity_floor = float(positivity_floor)

    @staticmethod
    def uniform_coordinates(refinement: int, *, dtype=None, device=None) -> Tensor:
        if refinement < 2:
            raise ValueError("refinement must be at least two")
        return (
            (torch.arange(refinement, dtype=dtype, device=device) + 0.5) / refinement
            - 0.5
        )

    def forward(self, cell_average: Tensor, subcell_coordinates: Tensor) -> Tensor:
        if cell_average.shape[-1] != 3:
            raise ValueError("cell_average must end in three conservative components")
        if cell_average.ndim < 2:
            raise ValueError("cell_average must include a cell dimension")
        if subcell_coordinates.ndim != 1:
            raise ValueError("subcell_coordinates must be one-dimensional")
        if not torch.allclose(
            torch.mean(subcell_coordinates),
            torch.zeros((), dtype=subcell_coordinates.dtype, device=subcell_coordinates.device),
            atol=10.0 * torch.finfo(subcell_coordinates.dtype).eps,
            rtol=0.0,
        ):
            raise ValueError("subcell coordinates must have zero mean")

        center_components = cell_average.movedim(-1, -2)
        primitive = conservative_to_primitive(cell_average, self.gamma)
        density = primitive[..., 0]
        velocity = primitive[..., 1]
        pressure = primitive[..., 2]
        sound_speed = torch.sqrt(
            self.gamma * torch.clamp(pressure, min=self.positivity_floor)
            / torch.clamp(density, min=self.positivity_floor)
        )
        context = torch.stack(
            (
                torch.tanh(torch.log(torch.clamp(density, min=self.positivity_floor))),
                torch.tanh(velocity / torch.clamp(sound_speed, min=1.0e-8)),
                torch.tanh(torch.log(torch.clamp(pressure, min=self.positivity_floor))),
            ),
            dim=-1,
        )
        q = subcell_coordinates.numel()
        xi = subcell_coordinates.reshape(*([1] * (cell_average.ndim - 1)), q)
        corrections = []
        stencils = []
        for component, network in enumerate(self.networks):
            center = center_components[..., component, :]
            fm2 = _outflow_shift(center, 2)
            fm1 = _outflow_shift(center, 1)
            fp1 = _outflow_shift(center, -1)
            fp2 = _outflow_shift(center, -2)
            differences = torch.stack(
                (fm2 - center, fm1 - center, fp1 - center, fp2 - center), dim=-1
            )
            scale = torch.clamp(torch.amax(torch.abs(differences), dim=-1), min=1.0e-6)
            magnitude = torch.abs(center) + scale + 1.0e-6
            left_difference = center - fm1
            right_difference = fp1 - center
            slope = minmod(
                2.0 * left_difference,
                0.5 * (left_difference + right_difference),
                2.0 * right_difference,
            )
            expanded_difference = differences.unsqueeze(-2).expand(*center.shape, q, 4)
            normalized_difference = expanded_difference / scale.unsqueeze(-1).unsqueeze(-1)
            center_feature = (center / magnitude).unsqueeze(-1).unsqueeze(-1).expand(
                *center.shape, q, 1
            )
            scale_feature = (scale / magnitude).unsqueeze(-1).unsqueeze(-1).expand(
                *center.shape, q, 1
            )
            xi_feature = (2.0 * xi).expand(*center.shape, q).unsqueeze(-1)
            context_feature = context.unsqueeze(-2).expand(*center.shape, q, 3)
            features = torch.cat(
                (
                    normalized_difference,
                    center_feature,
                    scale_feature,
                    xi_feature,
                    context_feature,
                ),
                dim=-1,
            )
            raw = torch.tanh(network(features.reshape(-1, 10))).reshape(*center.shape, q)
            zero_mean_raw = raw - torch.mean(raw, dim=-1, keepdim=True)
            corrections.append(
                slope.unsqueeze(-1) * xi
                + self.correction_scale * scale.unsqueeze(-1) * zero_mean_raw
            )
            stencils.append(torch.stack((fm2, fm1, center, fp1, fp2), dim=-1))

        correction = torch.stack(corrections, dim=-1)
        stencil = torch.stack(stencils, dim=-2)
        lower = torch.amin(stencil, dim=-1)
        upper = torch.amax(stencil, dim=-1)
        maximum_positive = torch.clamp(torch.amax(correction, dim=-2), min=1.0e-12)
        maximum_negative = torch.clamp(-torch.amin(correction, dim=-2), min=1.0e-12)
        bound_theta_components = torch.minimum(
            torch.ones_like(cell_average),
            torch.minimum(
                (upper - cell_average) / maximum_positive,
                (cell_average - lower) / maximum_negative,
            ),
        )
        bound_theta = torch.clamp(
            torch.amin(bound_theta_components, dim=-1), 0.0, 1.0
        )
        limited_correction = bound_theta.unsqueeze(-1).unsqueeze(-1) * correction
        positivity_theta = self._positivity_theta(cell_average, limited_correction)
        return cell_average.unsqueeze(-2) + positivity_theta.unsqueeze(-1).unsqueeze(
            -1
        ) * limited_correction

    def _positivity_theta(self, center: Tensor, correction: Tensor) -> Tensor:
        full_candidate = center.unsqueeze(-2) + correction
        full_density = full_candidate[..., 0]
        full_momentum = full_candidate[..., 1]
        full_energy = full_candidate[..., 2]
        full_pressure = (self.gamma - 1.0) * (
            full_energy
            - 0.5
            * full_momentum**2
            / torch.clamp(full_density, min=self.positivity_floor)
        )
        full_admissible = torch.all(
            (full_density >= self.positivity_floor)
            & (full_pressure >= self.positivity_floor),
            dim=-1,
        )
        if bool(torch.all(full_admissible).detach()):
            return torch.ones_like(center[..., 0])
        low = torch.zeros_like(center[..., 0])
        high = torch.ones_like(low)
        for _ in range(32):
            middle = 0.5 * (low + high)
            candidate = center.unsqueeze(-2) + middle.unsqueeze(-1).unsqueeze(-1) * correction
            density = candidate[..., 0]
            momentum = candidate[..., 1]
            energy = candidate[..., 2]
            safe_density = torch.clamp(density, min=self.positivity_floor)
            pressure = (self.gamma - 1.0) * (
                energy - 0.5 * momentum**2 / safe_density
            )
            admissible = torch.all(
                (density >= self.positivity_floor) & (pressure >= self.positivity_floor),
                dim=-1,
            )
            low = torch.where(admissible, middle, low)
            high = torch.where(admissible, high, middle)
        return low


class ConservativeEulerJumpBlendDKAN(nn.Module):
    """Learn one bounded blend between MC and an explicit conservative jump basis."""

    def __init__(
        self,
        hidden_width: int = 16,
        gamma: float = 1.4,
        positivity_floor: float = 1.0e-8,
        basis_type: str = "shock",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.positivity_floor = float(positivity_floor)
        if basis_type not in {"shock", "contact"}:
            raise ValueError("basis_type must be 'shock' or 'contact'")
        self.basis_type = basis_type
        self.fixed = ConservativeEulerDKANSubcell(
            hidden_width=8,
            correction_scale=0.0,
            gamma=gamma,
            positivity_floor=positivity_floor,
        )
        for parameter in self.fixed.parameters():
            parameter.requires_grad_(False)
        self.blend_network = DKAN([15, hidden_width, 1], grid_size=5, spline_order=3)
        output_layer = self.blend_network.layers[-1]
        nn.init.zeros_(output_layer.spline_coefficients)
        nn.init.zeros_(output_layer.dyt_weight)
        nn.init.constant_(output_layer.bias, -4.0)

    @staticmethod
    def uniform_coordinates(refinement: int, *, dtype=None, device=None) -> Tensor:
        return ConservativeEulerDKANSubcell.uniform_coordinates(
            refinement, dtype=dtype, device=device
        )

    def blend_features(self, cell_average: Tensor) -> Tensor:
        primitive = conservative_to_primitive(cell_average, self.gamma)
        differences = []
        for offset in (2, 1, -1, -2):
            shifted = torch.stack(
                [_outflow_shift(primitive[..., component], offset) for component in range(3)],
                dim=-1,
            )
            differences.append(shifted - primitive)
        stacked = torch.stack(differences, dim=-2)
        scale = torch.clamp(torch.amax(torch.abs(stacked), dim=-2), min=1.0e-6)
        normalized = (stacked / scale.unsqueeze(-2)).reshape(*cell_average.shape[:-1], 12)
        density, velocity, pressure = primitive.unbind(dim=-1)
        sound = torch.sqrt(
            self.gamma * torch.clamp(pressure, min=self.positivity_floor)
            / torch.clamp(density, min=self.positivity_floor)
        )
        context = torch.stack(
            (
                torch.tanh(torch.log(torch.clamp(density, min=self.positivity_floor))),
                torch.tanh(velocity / torch.clamp(sound, min=1.0e-8)),
                torch.tanh(torch.log(torch.clamp(pressure, min=self.positivity_floor))),
            ),
            dim=-1,
        )
        return torch.cat((normalized, context), dim=-1)

    def basis_profiles(
        self, cell_average: Tensor, subcell_coordinates: Tensor
    ) -> tuple[Tensor, Tensor]:
        fixed = self.fixed(cell_average, subcell_coordinates)
        refinement = subcell_coordinates.numel()
        if self.basis_type == "contact":
            primitive = conservative_to_primitive(cell_average, self.gamma)
            density = primitive[..., 0]
            left_density = _outflow_shift(density, 2)
            right_density = _outflow_shift(density, -2)
            density_direction = left_density - right_density
            safe_direction = torch.where(
                torch.abs(density_direction) >= 1.0e-12,
                density_direction,
                torch.ones_like(density_direction),
            )
            fraction = torch.clamp(
                (density - right_density) / safe_direction, 0.0, 1.0
            )
            index = torch.arange(
                refinement, dtype=cell_average.dtype, device=cell_average.device
            )
            left_weight = torch.clamp(
                fraction.unsqueeze(-1) * refinement - index, 0.0, 1.0
            )
            reconstructed_density = right_density.unsqueeze(-1) + left_weight * (
                left_density - right_density
            ).unsqueeze(-1)
            reconstructed_density = reconstructed_density + (
                density - torch.mean(reconstructed_density, dim=-1)
            ).unsqueeze(-1)
            contact_primitive = torch.stack(
                (
                    reconstructed_density,
                    primitive[..., 1].unsqueeze(-1).expand_as(reconstructed_density),
                    primitive[..., 2].unsqueeze(-1).expand_as(reconstructed_density),
                ),
                dim=-1,
            )
            return fixed, primitive_to_conservative(contact_primitive, self.gamma)
        left = torch.stack(
            [_outflow_shift(cell_average[..., component], 2) for component in range(3)],
            dim=-1,
        )
        right = torch.stack(
            [_outflow_shift(cell_average[..., component], -2) for component in range(3)],
            dim=-1,
        )
        direction = left - right
        fraction = torch.clamp(
            torch.sum((cell_average - right) * direction, dim=-1)
            / torch.clamp(torch.sum(direction**2, dim=-1), min=1.0e-12),
            0.0,
            1.0,
        )
        index = torch.arange(
            refinement, dtype=cell_average.dtype, device=cell_average.device
        )
        left_weight = torch.clamp(
            fraction.unsqueeze(-1) * refinement - index, 0.0, 1.0
        )
        jump = right.unsqueeze(-2) + left_weight.unsqueeze(-1) * direction.unsqueeze(-2)
        mixture_mean = fraction.unsqueeze(-1) * left + (1.0 - fraction).unsqueeze(-1) * right
        jump = jump + (cell_average - mixture_mean).unsqueeze(-2)
        return fixed, jump

    def forward(
        self,
        cell_average: Tensor,
        subcell_coordinates: Tensor,
        *,
        return_blend: bool = False,
        learned_blend_multiplier: float = 1.0,
        physics_sensor: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not 0.0 <= learned_blend_multiplier <= 1.0:
            raise ValueError("learned_blend_multiplier must lie in [0, 1]")
        fixed, jump = self.basis_profiles(cell_average, subcell_coordinates)
        features = self.blend_features(cell_average)
        blend = torch.sigmoid(
            self.blend_network(features.reshape(-1, 15)).reshape(*features.shape[:-1])
        )
        blend = learned_blend_multiplier * blend
        if physics_sensor:
            blend = blend * euler_discontinuity_sensor(
                cell_average,
                gamma=self.gamma,
                positivity_floor=self.positivity_floor,
            )
        reconstructed = self.reconstruct_from_blend(
            cell_average, subcell_coordinates, blend, basis=(fixed, jump)
        )
        if return_blend:
            return reconstructed, blend
        return reconstructed

    def reconstruct_from_blend(
        self,
        cell_average: Tensor,
        subcell_coordinates: Tensor,
        blend: Tensor,
        *,
        basis: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if blend.shape != cell_average.shape[:-1]:
            raise ValueError("blend must have one scalar per cell")
        fixed, jump = (
            self.basis_profiles(cell_average, subcell_coordinates)
            if basis is None
            else basis
        )
        candidate = fixed + blend.unsqueeze(-1).unsqueeze(-1) * (jump - fixed)
        correction = candidate - cell_average.unsqueeze(-2)

        stencils = []
        for component in range(3):
            center = cell_average[..., component]
            stencils.append(
                torch.stack(
                    (
                        _outflow_shift(center, 2),
                        _outflow_shift(center, 1),
                        center,
                        _outflow_shift(center, -1),
                        _outflow_shift(center, -2),
                    ),
                    dim=-1,
                )
            )
        stencil = torch.stack(stencils, dim=-2)
        lower = torch.amin(stencil, dim=-1)
        upper = torch.amax(stencil, dim=-1)
        maximum_positive = torch.clamp(torch.amax(correction, dim=-2), min=1.0e-12)
        maximum_negative = torch.clamp(-torch.amin(correction, dim=-2), min=1.0e-12)
        component_theta = torch.minimum(
            torch.ones_like(cell_average),
            torch.minimum(
                (upper - cell_average) / maximum_positive,
                (cell_average - lower) / maximum_negative,
            ),
        )
        bound_theta = torch.clamp(torch.amin(component_theta, dim=-1), 0.0, 1.0)
        correction = bound_theta.unsqueeze(-1).unsqueeze(-1) * correction
        positivity_theta = self.fixed._positivity_theta(cell_average, correction)
        reconstructed = cell_average.unsqueeze(-2) + positivity_theta.unsqueeze(
            -1
        ).unsqueeze(-1) * correction
        return reconstructed
