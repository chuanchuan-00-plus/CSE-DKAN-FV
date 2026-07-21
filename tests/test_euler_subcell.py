import unittest

import torch

from cse_dkan.euler_finite_volume import (
    conservative_to_primitive,
    primitive_to_conservative,
)
from cse_dkan.euler_subcell import (
    ConservativeEulerDKANSubcell,
    ConservativeEulerJumpBlendDKAN,
    euler_discontinuity_sensor,
    conservative_density_total_variation,
    density_tv_trust_projection,
)


class ConservativeEulerSubcellTests(unittest.TestCase):
    def test_cell_means_bounds_and_positivity(self):
        torch.manual_seed(31)
        primitive = torch.stack(
            (
                0.15 + torch.rand(3, 48, dtype=torch.float64),
                0.5 * torch.randn(3, 48, dtype=torch.float64),
                0.08 + torch.rand(3, 48, dtype=torch.float64),
            ),
            dim=-1,
        )
        state = primitive_to_conservative(primitive)
        model = ConservativeEulerDKANSubcell().double()
        coordinates = model.uniform_coordinates(8, dtype=state.dtype)
        reconstructed = model(state, coordinates)
        self.assertTrue(
            torch.allclose(reconstructed.mean(-2), state, atol=3.0e-12, rtol=0.0)
        )
        recovered = conservative_to_primitive(reconstructed)
        self.assertGreater(float(torch.min(recovered[..., 0]).detach()), 0.0)
        self.assertGreater(float(torch.min(recovered[..., 2]).detach()), 0.0)

    def test_constant_state_is_exact(self):
        primitive = torch.tensor([0.8, -0.15, 0.6], dtype=torch.float64).expand(32, 3)
        state = primitive_to_conservative(primitive)
        model = ConservativeEulerDKANSubcell().double()
        coordinates = model.uniform_coordinates(6, dtype=state.dtype)
        reconstructed = model(state, coordinates)
        self.assertTrue(
            torch.allclose(reconstructed, state.unsqueeze(-2).expand(-1, 6, -1))
        )

    def test_fixed_model_preserves_sod_cell_means(self):
        primitive = torch.cat(
            (
                torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64).expand(20, 3),
                torch.tensor([[0.125, 0.0, 0.1]], dtype=torch.float64).expand(20, 3),
            ),
            dim=0,
        )
        state = primitive_to_conservative(primitive)
        model = ConservativeEulerDKANSubcell(correction_scale=0.0).double()
        coordinates = model.uniform_coordinates(8, dtype=state.dtype)
        reconstructed = model(state, coordinates)
        self.assertTrue(
            torch.allclose(reconstructed.mean(-2), state, atol=3.0e-12, rtol=0.0)
        )

    def test_jump_blend_preserves_means_and_positivity(self):
        x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
        primitive = torch.where(
            (x < 0.55).unsqueeze(-1),
            torch.tensor([1.0, 0.1, 1.0], dtype=torch.float64),
            torch.tensor([0.15, 0.0, 0.12], dtype=torch.float64),
        )
        state = primitive_to_conservative(primitive)
        model = ConservativeEulerJumpBlendDKAN().double()
        coordinates = model.uniform_coordinates(8, dtype=state.dtype)
        reconstructed, blend = model(state, coordinates, return_blend=True)
        recovered = conservative_to_primitive(reconstructed)
        self.assertTrue(
            torch.allclose(reconstructed.mean(-2), state, atol=3.0e-12, rtol=0.0)
        )
        self.assertTrue(torch.all((blend >= 0.0) & (blend <= 1.0)))
        self.assertGreater(float(torch.min(recovered[..., 0]).detach()), 0.0)
        self.assertGreater(float(torch.min(recovered[..., 2]).detach()), 0.0)

    def test_zero_trust_multiplier_recovers_fixed_mc(self):
        x = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
        primitive = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.125, 0.0, 0.1], dtype=torch.float64),
        )
        state = primitive_to_conservative(primitive)
        learned = ConservativeEulerJumpBlendDKAN().double()
        fixed = ConservativeEulerDKANSubcell(correction_scale=0.0).double()
        coordinates = learned.uniform_coordinates(8, dtype=state.dtype)
        trusted = learned(
            state, coordinates, learned_blend_multiplier=0.0
        )
        baseline = fixed(state, coordinates)
        self.assertTrue(torch.allclose(trusted, baseline, atol=3.0e-12, rtol=0.0))

    def test_discontinuity_sensor_rejects_resolved_smooth_entropy_wave(self):
        cells = 128
        x = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        primitive = torch.stack(
            (
                1.0 + 0.2 * torch.sin(2.0 * torch.pi * x),
                torch.ones_like(x),
                torch.ones_like(x),
            ),
            dim=-1,
        )
        sensor = euler_discontinuity_sensor(primitive_to_conservative(primitive))
        self.assertEqual(float(torch.max(sensor)), 0.0)

    def test_discontinuity_sensor_activates_shock_and_contact(self):
        cells = 64
        x = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        shock = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.125, 0.0, 0.1], dtype=torch.float64),
        )
        contact = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.4, 0.35, 1.0], dtype=torch.float64),
            torch.tensor([0.2, 0.35, 1.0], dtype=torch.float64),
        )
        shock_sensor = euler_discontinuity_sensor(primitive_to_conservative(shock))
        contact_sensor = euler_discontinuity_sensor(primitive_to_conservative(contact))
        self.assertGreater(float(torch.max(shock_sensor)), 0.95)
        self.assertGreater(float(torch.max(contact_sensor)), 0.95)

    def test_physics_sensor_preserves_parent_means(self):
        cells = 64
        x = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        primitive = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.125, 0.0, 0.1], dtype=torch.float64),
        )
        state = primitive_to_conservative(primitive)
        model = ConservativeEulerJumpBlendDKAN().double()
        coordinates = model.uniform_coordinates(4, dtype=state.dtype)
        reconstructed = model(state, coordinates, physics_sensor=True)
        self.assertTrue(
            torch.allclose(reconstructed.mean(-2), state, atol=3.0e-12, rtol=0.0)
        )

    def test_density_tv_projection_preserves_means_positivity_and_budget(self):
        x = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
        primitive = torch.where(
            (x < 0.5).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.2, 0.0, 0.2], dtype=torch.float64),
        )
        state = primitive_to_conservative(primitive)
        fixed = state.unsqueeze(-2).expand(-1, 4, -1).clone()
        oscillation = torch.tensor([0.04, -0.04, 0.04, -0.04], dtype=torch.float64)
        candidate_primitive = conservative_to_primitive(fixed).clone()
        candidate_primitive[..., 0] += oscillation
        candidate = primitive_to_conservative(candidate_primitive)
        trusted, multiplier = density_tv_trust_projection(
            fixed, candidate, relative_budget=0.0
        )
        fixed_tv = conservative_density_total_variation(fixed)
        trusted_tv = conservative_density_total_variation(trusted)
        recovered = conservative_to_primitive(trusted)
        self.assertLessEqual(float(trusted_tv), float(fixed_tv) + 2.0e-10)
        self.assertTrue(torch.allclose(trusted.mean(-2), state, atol=3.0e-12, rtol=0.0))
        self.assertGreater(float(torch.min(recovered[..., 0])), 0.0)
        self.assertGreater(float(torch.min(recovered[..., 2])), 0.0)
        self.assertGreaterEqual(float(multiplier), 0.0)
        self.assertLessEqual(float(multiplier), 1.0)


if __name__ == "__main__":
    unittest.main()
