#!/usr/bin/env python3
"""
Unit/Integration test running the 50-image regression check against regression_before.pt.
Asserts predictions and box-level mAPs match bit-exact within 1e-6 tolerance.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.paths import REPO_ROOT


class TestRegressionSuite(unittest.TestCase):
    def test_regression_against_baseline(self):
        """Runs run_regression.py --compare regression_before.pt and asserts returncode 0."""
        baseline_file = REPO_ROOT / "regression_before.pt"
        self.assertTrue(baseline_file.exists(), f"Missing regression baseline: {baseline_file}")

        script_path = REPO_ROOT / "run_regression.py"
        cmd = [sys.executable, str(script_path), "--compare", str(baseline_file)]
        
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        self.assertEqual(proc.returncode, 0, "Regression suite failed to match regression_before.pt identically.")


if __name__ == "__main__":
    unittest.main()
