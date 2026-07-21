import unittest

import torch

from cse_dkan.fno import FNO1d, SpectralConv1d


class FNOTests(unittest.TestCase):
    def test_spectral_layer_shape_and_gradients(self):
        layer = SpectralConv1d(4, 7, 9)
        values = torch.randn(3, 4, 32, requires_grad=True)
        output = layer(values)
        self.assertEqual(output.shape, (3, 7, 32))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_operator_shape_and_determinism(self):
        torch.manual_seed(10)
        model = FNO1d(width=12, modes=8, layers=3)
        values = torch.randn(2, 64, 5)
        first = model(values)
        second = model(values)
        self.assertEqual(first.shape, (2, 64, 3))
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
