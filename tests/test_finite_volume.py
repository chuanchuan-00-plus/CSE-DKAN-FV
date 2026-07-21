import unittest

import torch

from cse_dkan.finite_volume import (
    ConservativeBurgersFV,
    DKANTVDLimiter,
    DKANWENO5Reconstruction,
    burgers_godunov_flux,
)


class GodunovFluxTests(unittest.TestCase):
    def test_flux_cases(self):
        left = torch.tensor([1.0, -2.0, -1.0, 2.0])
        right = torch.tensor([2.0, -1.0, 1.0, -1.0])
        flux = burgers_godunov_flux(left, right)
        expected = torch.tensor([0.5, 0.5, 0.0, 2.0])
        self.assertTrue(torch.allclose(flux, expected))


class ConservativeStepperTests(unittest.TestCase):
    def test_constant_state_is_preserved(self):
        solver = ConservativeBurgersFV(n_cells=64, limiter="tvd")
        initial = torch.full((64,), 0.37, dtype=torch.float64)
        final = solver.solve(initial, 0.2)
        self.assertLess(float(torch.max(torch.abs(final - initial))), 1.0e-12)

    def test_shared_flux_conserves_mass(self):
        torch.manual_seed(4)
        limiter = DKANTVDLimiter().double()
        solver = ConservativeBurgersFV(n_cells=96, limiter=limiter)
        initial = 0.2 + 0.3 * torch.randn(96, dtype=torch.float64)
        final = solver.solve(initial, 0.1)
        self.assertLess(abs(float((final.mean() - initial.mean()).detach())), 2.0e-12)

    def test_first_order_godunov_does_not_increase_quadratic_entropy(self):
        x = (torch.arange(128, dtype=torch.float64) + 0.5) / 64.0 - 1.0
        initial = 0.2 - torch.sin(torch.pi * x)
        solver = ConservativeBurgersFV(n_cells=128, limiter="first_order")
        final = solver.solve(initial, 0.2)
        self.assertLessEqual(float(torch.mean(final**2)), float(torch.mean(initial**2)) + 1.0e-12)

    def test_viscous_total_flux_preserves_mass(self):
        x = (torch.arange(96, dtype=torch.float64) + 0.5) / 48.0 - 1.0
        initial = 0.2 - torch.sin(torch.pi * x)
        solver = ConservativeBurgersFV(n_cells=96, limiter="tvd", viscosity=1.0e-3)
        final = solver.solve(initial, 0.2)
        self.assertLess(abs(float(final.mean() - initial.mean())), 2.0e-12)

    def test_dkan_weno_weights_are_bounded_and_conservative(self):
        torch.manual_seed(3)
        reconstruction = DKANWENO5Reconstruction().double()
        initial = 0.2 + 0.4 * torch.randn(2, 80, dtype=torch.float64)
        left, right = reconstruction(initial)
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.isfinite(right).all())
        solver = ConservativeBurgersFV(n_cells=80, limiter=reconstruction)
        final = solver.solve(initial, 0.03)
        self.assertLess(
            float(torch.max(torch.abs(final.mean(-1) - initial.mean(-1))).detach()),
            2.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
