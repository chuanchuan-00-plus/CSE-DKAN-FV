import unittest

import torch
from torch import nn

from cse_dkan.losses import (
    RectangularControlVolumes,
    burgers_control_volume_balance,
    burgers_entropy_violation,
    burgers_viscous_entropy_violation,
    burgers_rankine_hugoniot_residual,
    burgers_strong_residual,
)
from cse_dkan.models import (
    AdaptiveFourierFeatures,
    DKAN,
    DKANLayer,
    MLP,
    PeriodicCoordinateMap,
    ShockExplicitBurgersAnsatz,
    trainable_parameter_count,
)


class AnalyticLinearSolution(nn.Module):
    def forward(self, points):
        return points[:, :1] - 0.5 * points[:, 1:2]


class ConstantSolution(nn.Module):
    def forward(self, points):
        return torch.ones_like(points[:, :1]) * 0.25


class StationaryShockPosition(nn.Module):
    def forward(self, times):
        return torch.ones_like(times)


class ModelTests(unittest.TestCase):
    def test_dkan_layer_shape_and_gradients(self):
        layer = DKANLayer(3, 4, grid_size=5, spline_order=3).double()
        x = torch.linspace(-0.9, 0.9, 30, dtype=torch.float64).reshape(10, 3).requires_grad_(True)
        y = layer(x)
        self.assertEqual(y.shape, (10, 4))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.spline_coefficients.grad).all())

    def test_models_have_finite_outputs(self):
        x = torch.randn(16, 2)
        models = [MLP([2, 16, 1]), DKAN([2, 8, 1]), DKAN([2, 8, 8, 1], embedding_frequencies=6)]
        for model in models:
            self.assertTrue(torch.isfinite(model(x)).all())
            self.assertGreater(trainable_parameter_count(model), 0)

    def test_adaptive_fourier_shape(self):
        embedding = AdaptiveFourierFeatures(2, 7, 5)
        self.assertEqual(embedding(torch.randn(11, 2)).shape, (11, 5))

    def test_shock_explicit_ansatz_has_exact_initial_condition(self):
        coordinate_map = PeriodicCoordinateMap(-1.0, 1.0, 0.6)
        model = ShockExplicitBurgersAnsatz(
            MLP([3, 8, 1]), MLP([3, 8, 1]), coordinate_map, final_time=0.6
        )
        x = torch.linspace(-1.0, 1.0, 33).reshape(-1, 1)
        points = torch.cat((x, torch.zeros_like(x)), dim=-1)
        expected = 0.2 - torch.sin(torch.pi * x)
        self.assertTrue(torch.allclose(model(points), expected, atol=1.0e-7, rtol=1.0e-7))

    def test_shock_trajectory_is_anchored(self):
        coordinate_map = PeriodicCoordinateMap(-1.0, 1.0, 0.6)
        model = ShockExplicitBurgersAnsatz(
            MLP([3, 4, 1]),
            MLP([3, 4, 1]),
            coordinate_map,
            final_time=0.6,
            shock_origin=0.1,
        )
        self.assertAlmostEqual(float(model.shock_position(torch.zeros((1, 1))).detach()), 0.1, places=7)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.volumes = RectangularControlVolumes(
            centers=torch.tensor([[0.4, 0.2], [1.2, 0.4]]),
            half_width_x=torch.tensor([0.1, 0.15]),
            half_width_t=torch.tensor([0.08, 0.05]),
        )

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_constant_solution_satisfies_burgers_constraints(self):
        model = ConstantSolution()
        points = torch.rand(20, 2)
        strong = burgers_strong_residual(model, points)
        balance = burgers_control_volume_balance(model, self.volumes)
        entropy = burgers_entropy_violation(model, self.volumes)
        viscous_entropy = burgers_viscous_entropy_violation(model, self.volumes, 1.0e-2)
        self.assertLess(strong.abs().max().item(), 1.0e-12)
        self.assertLess(balance.abs().max().item(), 1.0e-12)
        self.assertLess(entropy.abs().max().item(), 1.0e-12)
        self.assertLess(viscous_entropy.abs().max().item(), 1.0e-12)

    def test_stationary_symmetric_burgers_shock_satisfies_rh(self):
        class SymmetricShock(nn.Module):
            def forward(self, points):
                return -torch.tanh(80.0 * (points[:, :1] - 1.0))

        times = torch.linspace(0.1, 0.8, 8)
        residual = burgers_rankine_hugoniot_residual(
            SymmetricShock(), StationaryShockPosition(), times, offset=0.08
        )
        self.assertLess(residual.abs().max().item(), 1.0e-12)

    def test_nonsolution_has_nonzero_strong_residual(self):
        points = torch.rand(32, 2)
        residual = burgers_strong_residual(AnalyticLinearSolution(), points)
        self.assertGreater(residual.abs().mean().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
