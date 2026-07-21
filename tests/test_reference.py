import unittest

import numpy as np

from cse_dkan.metrics import (
    gradient_equivalent_width,
    normalized_lp,
    periodic_mass,
    shock_location_from_gradient,
    shock_width_10_90,
    total_variation_excess,
)
from cse_dkan.reference import BurgersWENO, ExactSodSolver, PrimitiveState


class BurgersReferenceTests(unittest.TestCase):
    def test_constant_state_is_preserved(self):
        solver = BurgersWENO(n_cells=128)
        result = solver.solve(lambda x: np.full_like(x, 0.37), final_time=0.2)[0.2]
        self.assertLess(np.max(np.abs(result - 0.37)), 1.0e-12)

    def test_periodic_mass_is_conserved(self):
        solver = BurgersWENO(n_cells=256)
        initial = lambda x: -np.sin(np.pi * (x - 1.0))
        u0 = initial(solver.x)
        result = solver.solve(initial, final_time=0.2)[0.2]
        self.assertLess(abs(periodic_mass(result, 2.0) - periodic_mass(u0, 2.0)), 2.0e-12)

    def test_viscous_constant_state_is_preserved(self):
        solver = BurgersWENO(n_cells=96, viscosity=1.0e-2)
        result = solver.solve(lambda x: np.full_like(x, -0.2), final_time=0.05)[0.05]
        self.assertLess(np.max(np.abs(result + 0.2)), 1.0e-12)

    def test_smooth_solution_converges_before_shock(self):
        def initial(x):
            return 0.5 + 0.25 * np.sin(2.0 * np.pi * x)

        def exact(x, time):
            foot = (x - 0.5 * time) % 1.0
            for _ in range(20):
                value = initial(foot)
                residual = foot + time * value - x
                residual -= np.round(residual)
                jacobian = 1.0 + 0.5 * np.pi * time * np.cos(2.0 * np.pi * foot)
                foot = (foot - residual / jacobian) % 1.0
            return initial(foot)

        errors = []
        for cells in (64, 128):
            solver = BurgersWENO(x_left=0.0, x_right=1.0, n_cells=cells)
            numerical = solver.solve(initial, 0.2)[0.2]
            errors.append(float(np.sqrt(np.mean((numerical - exact(solver.x, 0.2)) ** 2))))
        self.assertLess(errors[1], errors[0] / 4.0)


class SodReferenceTests(unittest.TestCase):
    def test_standard_star_state(self):
        solver = ExactSodSolver()
        self.assertAlmostEqual(solver.p_star, 0.30313, places=4)
        self.assertAlmostEqual(solver.u_star, 0.92745, places=4)

    def test_solution_is_positive_and_has_expected_limits(self):
        solver = ExactSodSolver()
        x = np.linspace(0.0, 1.0, 1001)
        rho, velocity, pressure = solver.sample(x, 0.2)
        self.assertTrue(np.all(rho > 0.0))
        self.assertTrue(np.all(pressure > 0.0))
        self.assertAlmostEqual(rho[0], 1.0, places=12)
        self.assertAlmostEqual(velocity[0], 0.0, places=12)
        self.assertAlmostEqual(pressure[-1], 0.1, places=12)

    def test_lax_problem_can_be_constructed(self):
        solver = ExactSodSolver(
            left=PrimitiveState(0.445, 0.689, 3.528),
            right=PrimitiveState(0.5, 0.0, 0.571),
        )
        rho, velocity, pressure = solver.sample(np.linspace(0.0, 1.0, 101), 0.14)
        self.assertTrue(np.all(np.isfinite(rho)))
        self.assertTrue(np.all(pressure > 0.0))
        self.assertGreater(np.max(velocity), 1.0)


class MetricTests(unittest.TestCase):
    def test_identity_metrics(self):
        values = np.array([0.0, 1.0, 0.5])
        self.assertEqual(normalized_lp(values, values), 0.0)
        self.assertEqual(total_variation_excess(values, values), 0.0)

    def test_linear_transition_width(self):
        x = np.linspace(0.0, 1.0, 1001)
        profile = 1.0 - x
        width = shock_width_10_90(x, profile, left_state=1.0, right_state=0.0)
        self.assertAlmostEqual(width, 0.8, places=12)
        self.assertAlmostEqual(gradient_equivalent_width(x, profile, 1.0, 0.0), 1.0, places=12)

        localized = 1.0 - 0.5 * (1.0 + np.tanh((x - 0.5) / 0.02))
        self.assertAlmostEqual(shock_location_from_gradient(x, localized), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
