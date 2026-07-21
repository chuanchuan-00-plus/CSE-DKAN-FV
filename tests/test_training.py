import unittest

import torch

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.models import (
    ViscousLayerConservativeSpectralBurgersAnsatz,
    trainable_parameter_count,
)
from cse_dkan.training import BurgersTrainingConfig, _physics_loss, build_burgers_model


class TrainingTests(unittest.TestCase):
    def test_parameter_budgets_are_matched(self):
        problem = PeriodicBurgersProblem()
        device = torch.device("cpu")
        counts = {
            variant: trainable_parameter_count(build_burgers_model(variant, problem, device, torch.float32))
            for variant in (
                "mlp_pinn", "gw_pinn", "av_pinn", "ci_pinn",
                "dkan_pinn", "cw_dkan", "cse_dkan",
            )
        }
        reference = counts["mlp_pinn"]
        for count in counts.values():
            self.assertLessEqual(abs(count - reference) / reference, 0.05)

    def test_each_variant_builds_a_finite_training_loss(self):
        problem = PeriodicBurgersProblem()
        device = torch.device("cpu")
        for variant in (
            "mlp_pinn", "gw_pinn", "av_pinn", "ci_pinn",
            "dkan_pinn", "cw_dkan", "cse_dkan",
        ):
            config = BurgersTrainingConfig(
                variant=variant,
                batch_interior=16,
                batch_boundary=4,
                batch_control_volumes=2,
                batch_rh=2,
                device="cpu",
            )
            model = build_burgers_model(variant, problem, device, torch.float32)
            losses = _physics_loss(model, config, problem, device, torch.float32, step=1)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_cse_spectral_ansatz_conserves_periodic_mass_by_construction(self):
        problem = PeriodicBurgersProblem()
        model = build_burgers_model("cse_dkan", problem, torch.device("cpu"), torch.float64)
        x = torch.linspace(problem.x_left, problem.x_right, 4097, dtype=torch.float64)[:-1]
        for time in (0.1, 0.35, 0.6):
            points = torch.stack((x, torch.full_like(x, time)), dim=-1)
            predicted_mean = torch.mean(model(points))
            self.assertAlmostEqual(float(predicted_mean.detach()), problem.mean, places=6)

    def test_viscous_cse_uses_positive_physical_layer_and_conserves_mass(self):
        problem = PeriodicBurgersProblem(viscosity=1.0e-3)
        model = build_burgers_model(
            "cse_dkan_viscous", problem, torch.device("cpu"), torch.float64
        )
        self.assertIsInstance(model, ViscousLayerConservativeSpectralBurgersAnsatz)
        times = torch.tensor([[0.35], [0.6]], dtype=torch.float64)
        widths = model.physical_gate_width(times)
        self.assertTrue(torch.all(widths > 0.0))
        self.assertTrue(torch.all(widths <= model.maximum_layer_width))
        x = torch.linspace(problem.x_left, problem.x_right, 4097, dtype=torch.float64)[:-1]
        initial_points = torch.stack((x, torch.zeros_like(x)), dim=-1)
        self.assertTrue(
            torch.allclose(
                model(initial_points), problem.initial_torch(x.reshape(-1, 1)), atol=1.0e-7
            )
        )
        points = torch.stack((x, torch.full_like(x, 0.6)), dim=-1)
        self.assertAlmostEqual(float(torch.mean(model(points)).detach()), problem.mean, places=6)
        config = BurgersTrainingConfig(
            variant="cse_dkan_viscous",
            viscosity=problem.viscosity,
            batch_interior=16,
            batch_boundary=4,
            batch_control_volumes=2,
            batch_rh=2,
            device="cpu",
        )
        losses = _physics_loss(model, config, problem, torch.device("cpu"), torch.float64)
        self.assertIn("characteristic_trace", losses)
        self.assertNotIn("rh", losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))


if __name__ == "__main__":
    unittest.main()
