import unittest

import torch

from cse_dkan.euler_training import EulerTrainingConfig, _losses, build_euler_model
from cse_dkan.models import trainable_parameter_count


class EulerTrainingTests(unittest.TestCase):
    def test_parameter_budgets_match(self):
        device = torch.device("cpu")
        mlp = trainable_parameter_count(build_euler_model("mlp_pinn", device, torch.float32))
        for variant in (
            "gw_pinn", "av_pinn", "ci_pinn",
            "dkan_pinn", "cw_dkan", "cse_dkan",
        ):
            count = trainable_parameter_count(build_euler_model(variant, device, torch.float32))
            self.assertLessEqual(abs(count - mlp) / mlp, 0.05)

    def test_all_variants_have_finite_loss(self):
        device = torch.device("cpu")
        for variant in (
            "mlp_pinn", "gw_pinn", "av_pinn", "ci_pinn",
            "dkan_pinn", "cw_dkan", "cse_dkan",
        ):
            config = EulerTrainingConfig(
                variant=variant,
                steps=2,
                batch_interior=8,
                batch_initial=8,
                batch_boundary=4,
                batch_control_volumes=2,
                device="cpu",
            )
            model = build_euler_model(variant, device, torch.float32)
            losses = _losses(model, config, 1, device, torch.float32)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))


if __name__ == "__main__":
    unittest.main()
