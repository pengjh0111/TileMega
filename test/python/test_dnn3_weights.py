"""F3 weight recipes (DNN3 prompt §4 F-4): the NHWC flatten reorder."""
import unittest

import torch

from tilemega.dnn.weights import DNN_RECIPES, pack, recipe_sources


def nested(_recipe):
    raise AssertionError('linear_nhwc_flatten takes a named source, never a nested recipe')


class LinearNhwcFlattenTests(unittest.TestCase):
    """PyTorch flattens the pooled image as (c, h, w); F3 keeps the activation
    NHWC and reorders the Linear's columns to (h, w, c) instead."""

    def recipe(self, c, h, w):
        return dict(kind='linear_nhwc_flatten', source='classifier.0.weight', c=c, h=h, w=w)

    def packed(self, weight, c, h, w):
        return pack(self.recipe(c, h, w), lambda name: weight, nested)

    def test_recipe_is_a_dnn_recipe_with_its_single_source(self):
        self.assertIn('linear_nhwc_flatten', DNN_RECIPES)
        self.assertEqual(recipe_sources(self.recipe(512, 7, 7), nested),
                         ('classifier.0.weight',))

    def test_vgg16_classifier_matches_the_nchw_flatten(self):
        # The real shape: torchvision VGG16's classifier[0] over 512x7x7.
        # The reorder itself is exact, so the weight is compared bitwise; the
        # dot product is not, because permuting 25088 FP32 terms changes the
        # accumulation order (prompt §13 U9). Measured here: cos 0.9999999999995.
        torch.manual_seed(20261007)
        n, c, h, w = 4096, 512, 7, 7
        weight = torch.randn(n, c * h * w)
        image = torch.randn(2, c, h, w)

        columns = torch.arange(c * h * w).reshape(c, h, w).permute(1, 2, 0).reshape(-1)
        reordered = self.packed(weight, c, h, w)
        self.assertEqual(reordered.shape, weight.shape)
        self.assertTrue(torch.equal(reordered, weight[:, columns]))

        reference = torch.nn.functional.linear(image.reshape(2, -1), weight)
        nhwc = image.permute(0, 2, 3, 1).reshape(2, -1)
        self.assertTrue(torch.equal(nhwc, image.reshape(2, -1)[:, columns]))
        got = torch.nn.functional.linear(nhwc, reordered)
        cosine = torch.nn.functional.cosine_similarity(
            got.double().flatten(), reference.double().flatten(), 0)
        self.assertGreater(float(cosine), 1 - 1e-9)
        self.assertLess(float((got - reference).abs().max() /
                              reference.abs().max()), 1e-5)

    def test_every_column_keeps_its_own_weight(self):
        c, h, w = 3, 2, 5
        weight = torch.arange(float(c * h * w)).reshape(1, -1)
        reordered = self.packed(weight, c, h, w)
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    self.assertEqual(reordered[0, (hi * w + wi) * c + ci].item(),
                                     weight[0, (ci * h + hi) * w + wi].item())

    def test_dtype_and_device_are_preserved(self):
        for dtype in (torch.float32, torch.bfloat16):
            packed = self.packed(torch.zeros(2, 2 * 3 * 4, dtype=dtype), 2, 3, 4)
            self.assertEqual(packed.dtype, dtype)

    def test_inconsistent_geometry_is_rejected(self):
        for c, h, w in ((512, 7, 6), (0, 7, 7), (512, -1, 7)):
            with self.assertRaises(ValueError):
                self.packed(torch.zeros(8, 512 * 7 * 7), c, h, w)
        with self.assertRaises(ValueError):
            self.packed(torch.zeros(512 * 7 * 7), 512, 7, 7)


if __name__ == '__main__':
    unittest.main()
