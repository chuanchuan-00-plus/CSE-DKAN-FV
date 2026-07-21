"""MLP, adaptive Fourier embedding, and discontinuity-aware KAN layers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        widths: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must include input and output dimensions")
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            if weight_norm:
                linear = nn.utils.parametrizations.weight_norm(linear)
            layers.append(linear)
            if index < len(widths) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class PositivePrimitiveWrapper(nn.Module):
    """Map three unconstrained network outputs to positive density/pressure."""

    def __init__(self, network: nn.Module, density_floor: float = 1.0e-6, pressure_floor: float = 1.0e-6) -> None:
        super().__init__()
        self.network = network
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)

    def forward(self, points: Tensor) -> Tensor:
        raw = self.network(points)
        if raw.shape[-1] != 3:
            raise ValueError("primitive network must return three channels")
        density = self.density_floor + F.softplus(raw[:, :1])
        velocity = raw[:, 1:2]
        pressure = self.pressure_floor + F.softplus(raw[:, 2:3])
        return torch.cat((density, velocity, pressure), dim=-1)


class AffineCoordinateMap(nn.Module):
    """Map a rectangular physical domain to [-1, 1]^2."""

    def __init__(self, x_left: float, x_right: float, final_time: float) -> None:
        super().__init__()
        if x_right <= x_left or final_time <= 0.0:
            raise ValueError("invalid space-time domain")
        self.x_left = float(x_left)
        self.x_right = float(x_right)
        self.final_time = float(final_time)

    def forward(self, points: Tensor) -> Tensor:
        x = 2.0 * (points[:, :1] - self.x_left) / (self.x_right - self.x_left) - 1.0
        t = 2.0 * points[:, 1:2] / self.final_time - 1.0
        return torch.cat((x, t), dim=-1)


class AdaptiveFourierFeatures(nn.Module):
    """KAF-style low/high-frequency embedding with trainable frequencies."""

    def __init__(self, input_dim: int, frequencies: int, output_dim: int, sigma: float = 1.0) -> None:
        super().__init__()
        if min(input_dim, frequencies, output_dim) <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.frequency_matrix = nn.Parameter(torch.randn(frequencies, input_dim) * sigma)
        self.low = nn.Linear(input_dim, output_dim)
        self.high = nn.Linear(2 * frequencies, output_dim, bias=False)
        nn.init.xavier_normal_(self.low.weight)
        nn.init.zeros_(self.low.bias)
        nn.init.normal_(self.high.weight, mean=0.0, std=1.0e-3)

    def forward(self, inputs: Tensor) -> Tensor:
        phase = 2.0 * math.pi * (inputs @ self.frequency_matrix.T)
        fourier = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.low(F.gelu(inputs)) + self.high(fourier)


class DKANLayer(nn.Module):
    """Edge-wise Dynamic-Tanh plus B-spline KAN layer.

    Every input-output edge carries one trainable DyT atom and one trainable
    B-spline expansion, following the structural equation in the source paper.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, grid_size) <= 0 or spline_order < 0:
            raise ValueError("invalid DKAN layer dimensions")
        if grid_range[1] <= grid_range[0]:
            raise ValueError("grid_range must be increasing")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step
        knots = knots + grid_range[0]
        self.register_buffer("knots", knots)
        self.n_basis = grid_size + spline_order

        self.spline_coefficients = nn.Parameter(torch.empty(output_dim, input_dim, self.n_basis))
        self.spline_scale = nn.Parameter(torch.ones(output_dim, input_dim))
        self.dyt_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.dyt_raw_slope = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.dyt_center = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.spline_coefficients, std=0.05 / math.sqrt(max(input_dim, 1)))
        nn.init.xavier_uniform_(self.dyt_weight)

    def b_splines(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}) inputs")
        knots = self.knots.to(dtype=inputs.dtype, device=inputs.device)
        x = inputs.unsqueeze(-1)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(inputs.dtype)
        right_endpoint = torch.isclose(x, knots[-1])
        if right_endpoint.any():
            basis[..., -1] = torch.where(right_endpoint.squeeze(-1), torch.ones_like(basis[..., -1]), basis[..., -1])

        for degree in range(1, self.spline_order + 1):
            count = basis.shape[-1] - 1
            left_denominator = knots[degree : degree + count] - knots[:count]
            right_denominator = knots[degree + 1 : degree + 1 + count] - knots[1 : 1 + count]
            left = (x - knots[:count]) / left_denominator.clamp_min(torch.finfo(inputs.dtype).eps)
            right = (knots[degree + 1 : degree + 1 + count] - x) / right_denominator.clamp_min(
                torch.finfo(inputs.dtype).eps
            )
            basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
        if basis.shape[-1] != self.n_basis:
            raise RuntimeError(f"internal spline basis mismatch: {basis.shape[-1]} != {self.n_basis}")
        return basis

    def forward(self, inputs: Tensor) -> Tensor:
        basis = self.b_splines(inputs)
        spline = torch.einsum("bik,oik->bo", basis, self.spline_coefficients * self.spline_scale.unsqueeze(-1))
        slopes = F.softplus(self.dyt_raw_slope) + 1.0e-3
        dyt = torch.tanh(slopes.unsqueeze(0) * (inputs.unsqueeze(1) - self.dyt_center.unsqueeze(0)))
        dyt = torch.sum(dyt * self.dyt_weight.unsqueeze(0), dim=-1)
        return spline + dyt + self.bias


class DKAN(nn.Module):
    def __init__(
        self,
        widths: Sequence[int],
        grid_size: int = 5,
        spline_order: int = 3,
        embedding_frequencies: int | None = None,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must include input and output dimensions")
        self.embedding: nn.Module | None = None
        layer_widths = list(widths)
        if embedding_frequencies is not None:
            self.embedding = AdaptiveFourierFeatures(widths[0], embedding_frequencies, widths[1])
            layer_widths = [widths[1], *widths[2:]]
        self.layers = nn.ModuleList(
            DKANLayer(a, b, grid_size=grid_size, spline_order=spline_order)
            for a, b in zip(layer_widths[:-1], layer_widths[1:])
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.embedding(inputs) if self.embedding is not None else inputs
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class PeriodicCoordinateMap(nn.Module):
    """Map physical (x, t) coordinates to periodic, normalized features."""

    def __init__(self, x_left: float, x_right: float, final_time: float) -> None:
        super().__init__()
        if x_right <= x_left or final_time <= 0.0:
            raise ValueError("invalid space-time domain")
        self.x_left = float(x_left)
        self.domain_length = float(x_right - x_left)
        self.final_time = float(final_time)

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (batch, 2)")
        phase = 2.0 * math.pi * (points[:, :1] - self.x_left) / self.domain_length
        normalized_time = 2.0 * points[:, 1:2] / self.final_time - 1.0
        return torch.cat((torch.sin(phase), torch.cos(phase), normalized_time), dim=-1)


class BurgersInitialConditionAnsatz(nn.Module):
    """Impose a sinusoidal Burgers initial condition exactly."""

    def __init__(
        self,
        correction: nn.Module,
        coordinate_map: PeriodicCoordinateMap,
        mean: float = 0.2,
        amplitude: float = 1.0,
        origin: float = 0.0,
    ) -> None:
        super().__init__()
        self.correction = correction
        self.coordinate_map = coordinate_map
        self.mean = float(mean)
        self.amplitude = float(amplitude)
        self.origin = float(origin)

    def initial_condition(self, x: Tensor) -> Tensor:
        return self.mean - self.amplitude * torch.sin(math.pi * (x - self.origin))

    def forward(self, points: Tensor) -> Tensor:
        initial = self.initial_condition(points[:, :1])
        correction = self.correction(self.coordinate_map(points))
        return initial + points[:, 1:2] * correction


class PolynomialShockTrajectory(nn.Module):
    """A low-complexity learned shock curve anchored at its formation point."""

    def __init__(self, origin: float, initial_speed: float, degree: int = 3) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be at least one")
        coefficients = torch.zeros(degree)
        coefficients[0] = float(initial_speed)
        self.coefficients = nn.Parameter(coefficients)
        self.origin = float(origin)

    def forward(self, times: Tensor) -> Tensor:
        result = torch.zeros_like(times) + self.origin
        power = times
        for coefficient in self.coefficients:
            result = result + coefficient * power
            power = power * times
        return result


class ShockExplicitBurgersAnsatz(nn.Module):
    """Two smooth DKAN experts separated by a learned space-time shock manifold."""

    def __init__(
        self,
        left_correction: nn.Module,
        right_correction: nn.Module,
        coordinate_map: PeriodicCoordinateMap,
        final_time: float,
        shock_origin: float = 0.0,
        initial_shock_speed: float = 0.2,
        initial_onset_time: float = 1.0 / math.pi,
        onset_width: float = 0.025,
        gate_width: float = 0.04,
        mean: float = 0.2,
        amplitude: float = 1.0,
        origin: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_onset_time < final_time:
            raise ValueError("initial shock onset must lie inside the time domain")
        if onset_width <= 0.0 or gate_width <= 0.0:
            raise ValueError("transition widths must be positive")
        self.left_correction = left_correction
        self.right_correction = right_correction
        self.coordinate_map = coordinate_map
        self.trajectory = PolynomialShockTrajectory(shock_origin, initial_shock_speed, degree=3)
        onset_fraction = initial_onset_time / final_time
        self.raw_onset = nn.Parameter(torch.tensor(math.log(onset_fraction / (1.0 - onset_fraction))))
        self.final_time = float(final_time)
        self.onset_width = float(onset_width)
        self.gate_width = float(gate_width)
        self.mean = float(mean)
        self.amplitude = float(amplitude)
        self.origin = float(origin)

    def set_gate_width(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("gate width must be positive")
        self.gate_width = float(value)

    def onset_time(self) -> Tensor:
        return self.final_time * torch.sigmoid(self.raw_onset)

    def shock_position(self, times: Tensor) -> Tensor:
        return self.trajectory(times)

    def initial_condition(self, x: Tensor) -> Tensor:
        return self.mean - self.amplitude * torch.sin(math.pi * (x - self.origin))

    def branch_corrections(self, points: Tensor) -> tuple[Tensor, Tensor]:
        features = self.coordinate_map(points)
        return self.left_correction(features), self.right_correction(features)

    def forward(self, points: Tensor) -> Tensor:
        x, times = points[:, :1], points[:, 1:2]
        left, right = self.branch_corrections(points)
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        position = self.shock_position(times)
        gate = torch.sigmoid((x - position) / self.gate_width)
        smooth_correction = 0.5 * (left + right)
        split_correction = (1.0 - gate) * left + gate * right
        correction = (1.0 - activation) * smooth_correction + activation * split_correction
        return self.initial_condition(x) + times * correction


class LowRankShockBurgersAnsatz(nn.Module):
    """Shared DKAN trunk with an explicit low-rank jump/ridge basis.

    The trunk predicts a smooth correction and a low-rank jump amplitude.  The
    shock geometry is closed by characteristic feet, while the field retains
    enough freedom to satisfy the PDE away from the interface.
    """

    def __init__(
        self,
        correction_basis: nn.Module,
        coordinate_map: PeriodicCoordinateMap,
        final_time: float,
        shock_origin: float = 0.0,
        initial_shock_speed: float = 0.2,
        initial_onset_time: float = 1.0 / math.pi,
        onset_width: float = 0.025,
        gate_width: float = 0.04,
        mean: float = 0.2,
        amplitude: float = 1.0,
        origin: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_onset_time < final_time:
            raise ValueError("initial shock onset must lie inside the time domain")
        self.correction_basis = correction_basis
        self.coordinate_map = coordinate_map
        self.trajectory = PolynomialShockTrajectory(shock_origin, initial_shock_speed, degree=3)
        self.footpoint_center_coefficients = nn.Parameter(torch.zeros(2))
        inverse_softplus = math.log(math.exp(1.1) - 1.0)
        self.footpoint_separation_coefficients = nn.Parameter(
            torch.tensor([inverse_softplus, 0.0, 0.0])
        )
        fraction = initial_onset_time / final_time
        self.raw_onset = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction))))
        self.final_time = float(final_time)
        self.onset_width = float(onset_width)
        self.gate_width = float(gate_width)
        self.mean = float(mean)
        self.amplitude = float(amplitude)
        self.origin = float(origin)

    def set_gate_width(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("gate width must be positive")
        self.gate_width = float(value)

    def onset_time(self) -> Tensor:
        return self.final_time * torch.sigmoid(self.raw_onset)

    def shock_position(self, times: Tensor) -> Tensor:
        return self.trajectory(times)

    def initial_condition(self, x: Tensor) -> Tensor:
        return self.mean - self.amplitude * torch.sin(math.pi * (x - self.origin))

    def characteristic_footpoints(self, times: Tensor) -> tuple[Tensor, Tensor]:
        """Learned two-sided characteristic feet with fold-caustic scaling."""
        elapsed = F.relu(times - self.onset_time())
        center = torch.zeros_like(times) + self.origin
        power = elapsed
        for coefficient in self.footpoint_center_coefficients:
            center = center + coefficient * power
            power = power * elapsed
        raw_scale = torch.zeros_like(times)
        power = torch.ones_like(times)
        for coefficient in self.footpoint_separation_coefficients:
            raw_scale = raw_scale + coefficient * power
            power = power * elapsed
        separation = torch.sqrt(elapsed + 1.0e-8) * F.softplus(raw_scale)
        return center - separation, center + separation

    def _basis_at(self, points: Tensor) -> tuple[Tensor, Tensor]:
        basis = self.correction_basis(self.coordinate_map(points))
        if basis.shape[1] != 2:
            raise ValueError("low-rank correction basis must return two channels")
        return basis[:, :1], basis[:, 1:2]

    def interface_states(self, times: Tensor) -> tuple[Tensor, Tensor]:
        """Return exact left/right traces of the explicit jump basis."""
        position = self.shock_position(times)
        points = torch.cat((position, times), dim=-1)
        smooth, jump = self._basis_at(points)
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        initial = self.initial_condition(position)
        left = initial + times * (smooth + 0.5 * activation * jump)
        right = initial + times * (smooth - 0.5 * activation * jump)
        return left, right

    def forward(self, points: Tensor) -> Tensor:
        x, times = points[:, :1], points[:, 1:2]
        smooth, jump = self._basis_at(points)
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        position = self.shock_position(times)
        gate = torch.sigmoid((x - position) / self.gate_width)
        jump_basis = 0.5 - gate
        correction = smooth + activation * jump * jump_basis
        return self.initial_condition(x) + times * correction


class ConservativeSpectralShockBurgersAnsatz(LowRankShockBurgersAnsatz):
    """Mass-conservative spectral background plus a periodic zero-mean jump.

    The temporal DKAN emits Fourier coefficients and one jump coefficient.
    Every spatial basis has zero mean, so periodic Burgers mass is conserved by
    construction instead of being recovered through a soft penalty.
    """

    def __init__(self, *args, modes: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if modes < 1:
            raise ValueError("modes must be positive")
        self.modes = int(modes)

    def _coefficients(self, times: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized_time = 2.0 * times / self.final_time - 1.0
        values = self.correction_basis(normalized_time)
        expected = 2 * self.modes + 1
        if values.shape[1] != expected:
            raise ValueError(f"temporal coefficient network must return {expected} channels")
        return (
            values[:, : self.modes],
            values[:, self.modes : 2 * self.modes],
            values[:, -1:],
        )

    def _smooth_correction(self, x: Tensor, times: Tensor) -> Tensor:
        sine_coefficients, cosine_coefficients, _ = self._coefficients(times)
        length = self.coordinate_map.domain_length
        phase = 2.0 * math.pi * (x - self.coordinate_map.x_left) / length
        modes = torch.arange(1, self.modes + 1, dtype=x.dtype, device=x.device).reshape(1, -1)
        return torch.sum(
            sine_coefficients * torch.sin(phase * modes)
            + cosine_coefficients * torch.cos(phase * modes),
            dim=-1,
            keepdim=True,
        )

    def _jump_geometry(self, x: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        _, _, amplitude = self._coefficients(times)
        position = self.shock_position(times)
        length = self.coordinate_map.domain_length
        left = self.coordinate_map.x_left
        right = left + length
        scaled_left = (left - position) / self.gate_width
        scaled_right = (right - position) / self.gate_width
        mean_gate = self.gate_width * (F.softplus(scaled_right) - F.softplus(scaled_left)) / length
        gate = torch.sigmoid((x - position) / self.gate_width)
        ramp = (x - left) / length
        jump_basis = mean_gate - gate + ramp - 0.5
        return amplitude, jump_basis

    def interface_states(self, times: Tensor) -> tuple[Tensor, Tensor]:
        position = self.shock_position(times)
        smooth = self._smooth_correction(position, times)
        _, _, amplitude = self._coefficients(times)
        length = self.coordinate_map.domain_length
        left_boundary = self.coordinate_map.x_left
        right_boundary = left_boundary + length
        mean_gate = self.gate_width * (
            F.softplus((right_boundary - position) / self.gate_width)
            - F.softplus((left_boundary - position) / self.gate_width)
        ) / length
        center = mean_gate + (position - left_boundary) / length - 0.5
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        base = self.initial_condition(position) + times * smooth
        left = base + times * activation * amplitude * center
        right = base + times * activation * amplitude * (center - 1.0)
        return left, right

    def forward(self, points: Tensor) -> Tensor:
        x, times = points[:, :1], points[:, 1:2]
        smooth = self._smooth_correction(x, times)
        amplitude, jump_basis = self._jump_geometry(x, times)
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        return self.initial_condition(x) + times * (smooth + activation * amplitude * jump_basis)


class ViscousLayerConservativeSpectralBurgersAnsatz(ConservativeSpectralShockBurgersAnsatz):
    """Mass-conservative ansatz with a viscosity-calibrated travelling-wave layer.

    For a Burgers shock with jump ``delta_u``, the exact travelling wave is a
    logistic transition with scale ``2 * viscosity / |delta_u|``.  The outer
    jump is anchored by the two pre-trained characteristic feet, while DKAN may
    supply only a bounded relative correction.  The layer therefore starts
    broad and sharpens continuously as the shock forms without the degenerate
    zero-jump solution of a purely soft trace loss.
    """

    def __init__(
        self,
        *args,
        viscosity: float,
        minimum_layer_width: float | None = None,
        maximum_layer_width: float = 0.06,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if maximum_layer_width <= 0.0:
            raise ValueError("maximum layer width must be positive")
        minimum = 0.25 * viscosity if minimum_layer_width is None else minimum_layer_width
        if minimum <= 0.0 or minimum > maximum_layer_width:
            raise ValueError("invalid viscous layer width bounds")
        self.viscosity = float(viscosity)
        self.minimum_layer_width = float(minimum)
        self.maximum_layer_width = float(maximum_layer_width)
        self.gate_width = self.maximum_layer_width

    def set_gate_width(self, value: float) -> None:
        """Set only the broad-layer cap; physical scale remains viscosity driven."""
        if value <= 0.0:
            raise ValueError("gate width must be positive")
        self.maximum_layer_width = float(value)
        self.gate_width = float(value)

    def characteristic_outer_states(self, times: Tensor) -> tuple[Tensor, Tensor]:
        foot_left, foot_right = self.characteristic_footpoints(times)
        return self.initial_condition(foot_left), self.initial_condition(foot_right)

    def physical_jump(self, times: Tensor, correction: Tensor | None = None) -> Tensor:
        if correction is None:
            _, _, correction = self._coefficients(times)
        outer_left, outer_right = self.characteristic_outer_states(times)
        activation = torch.sigmoid((times - self.onset_time()) / self.onset_width)
        anchored_jump = activation * (outer_left - outer_right)
        return anchored_jump * (1.0 + 0.1 * torch.tanh(correction))

    def physical_gate_width(self, times: Tensor, correction: Tensor | None = None) -> Tensor:
        jump = torch.abs(self.physical_jump(times, correction))
        width = 2.0 * self.viscosity / torch.clamp(jump, min=1.0e-4)
        return torch.clamp(width, min=self.minimum_layer_width, max=self.maximum_layer_width)

    def _jump_geometry(self, x: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        _, _, correction = self._coefficients(times)
        jump = self.physical_jump(times, correction)
        position = self.shock_position(times)
        width = self.physical_gate_width(times, correction)
        length = self.coordinate_map.domain_length
        left = self.coordinate_map.x_left
        right = left + length
        scaled_left = (left - position) / width
        scaled_right = (right - position) / width
        mean_gate = width * (F.softplus(scaled_right) - F.softplus(scaled_left)) / length
        gate = torch.sigmoid((x - position) / width)
        ramp = (x - left) / length
        jump_basis = mean_gate - gate + ramp - 0.5
        return jump, jump_basis

    def interface_states(self, times: Tensor) -> tuple[Tensor, Tensor]:
        position = self.shock_position(times)
        smooth = self._smooth_correction(position, times)
        _, _, correction = self._coefficients(times)
        jump = self.physical_jump(times, correction)
        width = self.physical_gate_width(times, correction)
        length = self.coordinate_map.domain_length
        left_boundary = self.coordinate_map.x_left
        right_boundary = left_boundary + length
        mean_gate = width * (
            F.softplus((right_boundary - position) / width)
            - F.softplus((left_boundary - position) / width)
        ) / length
        center = mean_gate + (position - left_boundary) / length - 0.5
        base = self.initial_condition(position) + times * smooth
        left = base + jump * center
        right = base + jump * (center - 1.0)
        return left, right

    def forward(self, points: Tensor) -> Tensor:
        x, times = points[:, :1], points[:, 1:2]
        smooth = self._smooth_correction(x, times)
        jump, jump_basis = self._jump_geometry(x, times)
        return self.initial_condition(x) + times * smooth + jump * jump_basis


class EulerWaveExplicitAnsatz(nn.Module):
    """Positive primitive field with separate learned contact and shock curves."""

    def __init__(
        self,
        network: nn.Module,
        coordinate_map: AffineCoordinateMap,
        contact_speed: float,
        shock_speed: float,
        contact_width: float = 0.025,
        shock_width: float = 0.02,
        density_floor: float = 1.0e-6,
        pressure_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.network = network
        self.coordinate_map = coordinate_map
        self.contact_coefficients = nn.Parameter(torch.tensor([contact_speed, 0.0]))
        self.shock_coefficients = nn.Parameter(torch.tensor([shock_speed, 0.0]))
        self.register_buffer("_contact_width", torch.tensor(float(contact_width)))
        self.register_buffer("_shock_width", torch.tensor(float(shock_width)))
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)

    def set_gate_widths(self, contact_width: float, shock_width: float) -> None:
        if min(contact_width, shock_width) <= 0.0:
            raise ValueError("gate widths must be positive")
        self._contact_width.fill_(float(contact_width))
        self._shock_width.fill_(float(shock_width))

    @property
    def contact_width(self) -> float:
        return float(self._contact_width.detach())

    @property
    def shock_width(self) -> float:
        return float(self._shock_width.detach())

    @staticmethod
    def _trajectory(times: Tensor, coefficients: Tensor) -> Tensor:
        return 0.5 + coefficients[0] * times + coefficients[1] * times**2

    def contact_position(self, times: Tensor) -> Tensor:
        return self._trajectory(times, self.contact_coefficients)

    def shock_position(self, times: Tensor) -> Tensor:
        return self._trajectory(times, self.shock_coefficients)

    def _network_values(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        values = self.network(self.coordinate_map(points))
        if values.shape[-1] != 9:
            raise ValueError("Euler wave network must return nine channels")
        return torch.split(values, 3, dim=-1)

    def _primitive(self, raw: Tensor) -> Tensor:
        density = self.density_floor + F.softplus(raw[:, :1])
        velocity = raw[:, 1:2]
        pressure = self.pressure_floor + F.softplus(raw[:, 2:3])
        return torch.cat((density, velocity, pressure), dim=-1)

    def wave_traces(self, times: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        contact_points = torch.cat((self.contact_position(times), times), dim=-1)
        shock_points = torch.cat((self.shock_position(times), times), dim=-1)
        contact_base, contact_jump, _ = self._network_values(contact_points)
        shock_base, shock_contact_jump, shock_jump = self._network_values(shock_points)
        contact_left = self._primitive(contact_base)
        contact_right = self._primitive(contact_base + contact_jump)
        shock_left = self._primitive(shock_base + shock_contact_jump)
        shock_right = self._primitive(shock_base + shock_contact_jump + shock_jump)
        return contact_left, contact_right, shock_left, shock_right

    def forward(self, points: Tensor) -> Tensor:
        base, contact_jump, shock_jump = self._network_values(points)
        times, x = points[:, 1:2], points[:, :1]
        contact_gate = torch.sigmoid((x - self.contact_position(times)) / self.contact_width)
        shock_gate = torch.sigmoid((x - self.shock_position(times)) / self.shock_width)
        raw = base + contact_gate * contact_jump + shock_gate * shock_jump
        return self._primitive(raw)
