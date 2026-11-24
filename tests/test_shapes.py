#!/usr/bin/env python3
"""
Unit test asserting that all model checkpoints and cache tensors match audited shapes.
Checks:
- saved_models_v1.0
- saved_models_v2.0
- saved_models_v1.0_train90
- saved_models_v2.0_train90
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.checkpoint_loader import load_vynix_checkpoint


class TestModelShapes(unittest.TestCase):
    def test_v1_0_shapes(self):
        ckpt = load_vynix_checkpoint("v1.0", device="cpu", load_to_device=False)
        self.assertEqual(ckpt["cache_keys"].shape, torch.Size([19824, 1536]))
        self.assertEqual(ckpt["cache_values"].shape, torch.Size([19824, 600]))
        self.assertEqual(ckpt["num_entries"], 19824)
        self.assertAlmostEqual(ckpt["alpha"], 0.954850, places=4)
        self.assertAlmostEqual(ckpt["beta"], 5.558595, places=4)

    def test_v2_0_shapes(self):
        ckpt = load_vynix_checkpoint("v2.0", device="cpu", load_to_device=False)
        self.assertEqual(ckpt["cache_keys"].shape, torch.Size([20006, 1536]))
        self.assertEqual(ckpt["cache_values"].shape, torch.Size([20006, 600]))
        self.assertEqual(ckpt["num_entries"], 20006)
        self.assertAlmostEqual(ckpt["alpha"], 0.939244, places=4)
        self.assertAlmostEqual(ckpt["beta"], 5.593530, places=4)

    def test_v1_0_train90_shapes(self):
        try:
            ckpt = load_vynix_checkpoint("v1.0_train90", device="cpu", load_to_device=False)
        except FileNotFoundError as e:
            self.skipTest(f"v1.0_train90 checkpoint file not present in local workspace: {e}")
        self.assertEqual(ckpt["cache_keys"].shape, torch.Size([19579, 1536]))
        self.assertEqual(ckpt["cache_values"].shape, torch.Size([19579, 600]))
        self.assertEqual(ckpt["num_entries"], 19579)
        self.assertAlmostEqual(ckpt["alpha"], 0.954937, places=4)
        self.assertAlmostEqual(ckpt["beta"], 5.558463, places=4)

    def test_v2_0_train90_shapes(self):
        try:
            ckpt = load_vynix_checkpoint("v2.0_train90", device="cpu", load_to_device=False)
        except FileNotFoundError as e:
            self.skipTest(f"v2.0_train90 checkpoint file not present in local workspace: {e}")
        self.assertEqual(ckpt["cache_keys"].shape, torch.Size([19673, 1536]))
        self.assertEqual(ckpt["cache_values"].shape, torch.Size([19673, 600]))
        self.assertEqual(ckpt["num_entries"], 19673)
        self.assertAlmostEqual(ckpt["alpha"], 0.939605, places=4)
        self.assertAlmostEqual(ckpt["beta"], 5.592494, places=4)


if __name__ == "__main__":
    unittest.main()
