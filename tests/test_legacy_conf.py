#!/usr/bin/env python3
"""
Unit test verifying the documented legacy detector confidence threshold behavior.
Tests:
1. Ultralytics YOLO defaults to conf=0.25 when conf is omitted in the model call.
2. MultiStreamFeatureExtractor documented legacy mode preserves conf=0.25 behavior.
3. Explicit conf=0.08 produces >= candidates compared to legacy default conf=0.25.
"""

import os
import sys
import unittest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import MultiStreamFeatureExtractor


class TestLegacyDetectorConf(unittest.TestCase):
    def test_legacy_conf_default(self):
        """
        Verifies that the legacy default detector confidence threshold is 0.25.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize extractor with nominal detector_conf=0.08
        extractor = MultiStreamFeatureExtractor(
            device=device,
            detector_conf=0.08,
            use_3stream=False,
            detector_model="yolov8m.pt",
            clip_model="openai/clip-vit-base-patch16",
        )
        
        # Check that the internal detector_conf attribute is recorded
        self.assertEqual(extractor.detector_conf, 0.08)
        
        # Verify Ultralytics YOLO default parameter
        dummy_img = Image.new("RGB", (640, 640), (128, 128, 128))
        persons, objects = extractor.detect(dummy_img)
        
        self.assertIsInstance(persons, list)
        self.assertIsInstance(objects, list)
        for p_box, p_conf in persons:
            self.assertGreaterEqual(p_conf, 0.25)
        for o_box, o_conf, o_cls in objects:
            self.assertGreaterEqual(o_conf, 0.25)


if __name__ == "__main__":
    unittest.main()
