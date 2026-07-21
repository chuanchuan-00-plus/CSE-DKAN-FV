import unittest

import torch

from cse_dkan.euler_finite_volume import (
    EulerHLLCFV,
    conservative_to_primitive,
    euler_flux,
    hllc_flux,
    primitive_to_conservative,
)
from cse_dkan.euler_losses import euler_control_volume_balance, euler_strong_residual
from cse_dkan.losses import RectangularControlVolumes
from torch import nn


class EulerConversionTests(unittest.TestCase):
    def test_round_trip(self):
        primitive = torch.tensor([[1.0, 0.3, 1.2], [0.4, -0.2, 0.15]], dtype=torch.float64)
        recovered = conservative_to_primitive(primitive_to_conservative(primitive))
        self.assertTrue(torch.allclose(recovered, primitive, atol=1.0e-13, rtol=1.0e-13))

    def test_hllc_consistency(self):
        primitive = torch.tensor([[1.0, 0.25, 1.0]], dtype=torch.float64)
        state = primitive_to_conservative(primitive)
        self.assertTrue(torch.allclose(hllc_flux(state, state), euler_flux(state), atol=1.0e-12))


class EulerStepperTests(unittest.TestCase):
    def test_constant_state_is_preserved(self):
        solver = EulerHLLCFV(n_cells=64, boundary="periodic")
        primitive = torch.tensor([0.8, 0.2, 0.7], dtype=torch.float64).expand(64, 3)
        initial = primitive_to_conservative(primitive)
        final = solver.solve(initial, 0.1)
        self.assertLess(float(torch.max(torch.abs(final - initial))), 1.0e-12)

    def test_periodic_total_conservative_state_is_preserved(self):
        solver = EulerHLLCFV(n_cells=96, boundary="periodic", reconstruction="muscl_mc")
        x = (torch.arange(96, dtype=torch.float64) + 0.5) / 96.0
        primitive = torch.stack(
            (1.0 + 0.1 * torch.sin(2.0 * torch.pi * x), 0.1 * torch.cos(2.0 * torch.pi * x), torch.ones_like(x)),
            dim=-1,
        )
        initial = primitive_to_conservative(primitive)
        final = solver.solve(initial, 0.03)
        self.assertTrue(torch.allclose(final.mean(dim=0), initial.mean(dim=0), atol=2.0e-12, rtol=0.0))
        final_primitive = conservative_to_primitive(final)
        self.assertTrue(torch.all(final_primitive[:, 0] > 0.0))
        self.assertTrue(torch.all(final_primitive[:, 2] > 0.0))

    def test_muscl_sod_reconstruction_remains_positive(self):
        cells = 100
        x = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        primitive = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.125, 0.0, 0.1], dtype=torch.float64),
        )
        solver = EulerHLLCFV(
            n_cells=cells, boundary="outflow", reconstruction="muscl_mc", cfl=0.25
        )
        final = conservative_to_primitive(
            solver.solve(primitive_to_conservative(primitive), 0.05)
        )
        self.assertTrue(torch.all(final[:, 0] > 0.0))
        self.assertTrue(torch.all(final[:, 2] > 0.0))

    def test_reflecting_boundary_has_zero_mass_and_energy_flux(self):
        cells = 48
        solver = EulerHLLCFV(
            n_cells=cells, boundary="reflecting", reconstruction="muscl_mc", cfl=0.2
        )
        x = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        primitive = torch.stack(
            (
                1.0 + 0.05 * torch.cos(2.0 * torch.pi * x),
                0.1 * torch.sin(torch.pi * x),
                0.8 + 0.03 * torch.cos(2.0 * torch.pi * x),
            ),
            dim=-1,
        )
        initial = primitive_to_conservative(primitive)
        boundary_flux = solver.interface_flux(initial)[[0, -1]]
        self.assertLess(float(torch.max(torch.abs(boundary_flux[:, 0]))), 1.0e-12)
        self.assertLess(float(torch.max(torch.abs(boundary_flux[:, 2]))), 1.0e-12)
        final = solver.solve(initial, 0.02)
        self.assertTrue(
            torch.allclose(final[:, [0, 2]].sum(dim=0), initial[:, [0, 2]].sum(dim=0), atol=5.0e-11, rtol=0.0)
        )


class EulerPINNLossTests(unittest.TestCase):
    def test_constant_primitive_state_has_zero_strong_and_weak_residual(self):
        class ConstantPrimitive(nn.Module):
            def forward(self, points):
                value = torch.tensor([0.8, 0.2, 0.7], dtype=points.dtype, device=points.device)
                return points[:, :1] * 0.0 + value

        model = ConstantPrimitive()
        points = torch.rand(20, 2, dtype=torch.float64)
        volumes = RectangularControlVolumes(
            torch.tensor([[0.3, 0.1], [0.7, 0.15]], dtype=torch.float64),
            torch.tensor([[0.08], [0.06]], dtype=torch.float64),
            torch.tensor([[0.04], [0.03]], dtype=torch.float64),
        )
        self.assertLess(
            float(torch.max(torch.abs(euler_strong_residual(model, points))).detach()), 1.0e-12
        )
        self.assertLess(
            float(torch.max(torch.abs(euler_control_volume_balance(model, volumes)))), 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
