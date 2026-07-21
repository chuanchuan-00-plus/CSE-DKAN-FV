import unittest

import torch

from cse_dkan.subcell import ConservativeDKANSubcell


class ConservativeSubcellTests(unittest.TestCase):
    def test_reconstruction_preserves_every_cell_average_and_bounds(self):
        torch.manual_seed(8)
        model = ConservativeDKANSubcell().double()
        state = 0.2 + 0.4 * torch.randn(3, 40, dtype=torch.float64)
        coordinates = model.uniform_coordinates(8, dtype=state.dtype)
        reconstructed = model(state, coordinates, viscosity=1.0e-3, dx=0.05)
        self.assertTrue(torch.allclose(reconstructed.mean(-1), state, atol=2.0e-12, rtol=0.0))
        stencil = torch.stack(
            (
                torch.roll(state, 2, -1),
                torch.roll(state, 1, -1),
                state,
                torch.roll(state, -1, -1),
                torch.roll(state, -2, -1),
            ),
            dim=-1,
        )
        self.assertTrue(torch.all(reconstructed >= stencil.amin(-1).unsqueeze(-1) - 1.0e-12))
        self.assertTrue(torch.all(reconstructed <= stencil.amax(-1).unsqueeze(-1) + 1.0e-12))

    def test_constant_state_is_exact(self):
        model = ConservativeDKANSubcell().double()
        state = torch.full((24,), 0.37, dtype=torch.float64)
        coordinates = model.uniform_coordinates(6, dtype=state.dtype)
        reconstructed = model(state, coordinates, viscosity=0.0, dx=1.0 / 12.0)
        self.assertTrue(torch.allclose(reconstructed, torch.full_like(reconstructed, 0.37)))

    def test_shock_sensor_gate_recovers_fixed_mc_when_inactive(self):
        torch.manual_seed(19)
        learned = ConservativeDKANSubcell(correction_scale=0.5).double()
        fixed = ConservativeDKANSubcell(correction_scale=0.0).double()
        state = 0.2 + 0.4 * torch.randn(2, 32, dtype=torch.float64)
        coordinates = learned.uniform_coordinates(8, dtype=state.dtype)
        gated = learned(
            state,
            coordinates,
            viscosity=0.0,
            dx=1.0 / 16.0,
            shock_sensor_threshold=1.1,
        )
        baseline = fixed(state, coordinates, viscosity=0.0, dx=1.0 / 16.0)
        self.assertTrue(torch.allclose(gated, baseline, atol=2.0e-12, rtol=0.0))

    def test_zero_trust_multiplier_recovers_fixed_mc(self):
        torch.manual_seed(20)
        learned = ConservativeDKANSubcell(correction_scale=0.5).double()
        fixed = ConservativeDKANSubcell(correction_scale=0.0).double()
        state = 0.2 + 0.4 * torch.randn(32, dtype=torch.float64)
        coordinates = learned.uniform_coordinates(8, dtype=state.dtype)
        trusted = learned(
            state,
            coordinates,
            viscosity=1.0e-3,
            dx=1.0 / 16.0,
            learned_correction_multiplier=0.0,
        )
        baseline = fixed(state, coordinates, viscosity=1.0e-3, dx=1.0 / 16.0)
        self.assertTrue(torch.allclose(trusted, baseline, atol=2.0e-12, rtol=0.0))

    def test_shock_sensor_is_amplitude_invariant_and_dilated(self):
        state = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        scaled = 3.7 * state - 1.2
        sensor = ConservativeDKANSubcell.jump_sensor(
            state, threshold=0.3, dilation=1
        )
        scaled_sensor = ConservativeDKANSubcell.jump_sensor(
            scaled, threshold=0.3, dilation=1
        )
        self.assertTrue(torch.equal(sensor, scaled_sensor))
        self.assertEqual(int(torch.sum(sensor).item()), 6)

    def test_minimum_jump_disables_weak_discontinuity(self):
        state = torch.tensor([0.0, 0.0, 0.0, 0.8, 0.8, 0.8], dtype=torch.float64)
        sensor = ConservativeDKANSubcell.jump_sensor(
            state, threshold=0.3, minimum_jump=1.1, dilation=1
        )
        self.assertEqual(int(torch.sum(sensor).item()), 0)

    def test_resolved_viscous_transition_recovers_fixed_mc(self):
        torch.manual_seed(21)
        learned = ConservativeDKANSubcell(correction_scale=0.5).double()
        fixed = ConservativeDKANSubcell(correction_scale=0.0).double()
        state = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        coordinates = learned.uniform_coordinates(8, dtype=state.dtype)
        gated = learned(
            state,
            coordinates,
            viscosity=0.1,
            dx=0.1,
            shock_sensor_threshold=0.0,
            maximum_viscous_transition_cells=2.0,
        )
        baseline = fixed(state, coordinates, viscosity=0.1, dx=0.1)
        self.assertTrue(torch.allclose(gated, baseline, atol=2.0e-12, rtol=0.0))

    def test_tv_projection_preserves_averages_and_enforces_budget(self):
        model = ConservativeDKANSubcell().double()
        state = torch.tensor([0.0, 0.4, 1.0, 0.6, -0.2, -0.5], dtype=torch.float64)
        coordinates = model.uniform_coordinates(8, dtype=state.dtype)
        reconstructed = model(state, coordinates, viscosity=0.0, dx=1.0 / 3.0)
        projected = model.project_total_variation(reconstructed, state, relative_budget=0.01)
        self.assertTrue(torch.allclose(projected.mean(-1), state, atol=2.0e-12, rtol=0.0))
        flat = projected.reshape(-1)
        projected_tv = torch.sum(torch.abs(flat - torch.roll(flat, 1)))
        base_tv = torch.sum(torch.abs(state - torch.roll(state, 1)))
        self.assertLessEqual(
            float(projected_tv.detach()), 1.01 * float(base_tv) + 1.0e-12
        )
        absolute_projected = model.project_total_variation(
            reconstructed, state, absolute_budget=1.05 * base_tv
        )
        absolute_flat = absolute_projected.reshape(-1)
        absolute_tv = torch.sum(
            torch.abs(absolute_flat - torch.roll(absolute_flat, 1))
        )
        self.assertLessEqual(
            float(absolute_tv.detach()), 1.05 * float(base_tv) + 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
