"""
Unit Tests for Standard Box-Level HICO-DET Evaluator
===================================================
Tests matching logic, duplicate suppression, IoU thresholds,
and exact PR-AUC calculation against known mathematical ground truth.
"""

import sys
import os
import numpy as np

# Ensure import works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hico_evaluator import compute_box_iou, compute_continuous_ap, HICODetBoxEvaluator


def test_box_iou():
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [0.0, 0.0, 10.0, 10.0]
    assert abs(compute_box_iou(b1, b2) - 1.0) < 1e-6, "Self IoU must be 1.0"

    b3 = [5.0, 0.0, 15.0, 10.0]  # overlap 5x10=50, union 100+100-50=150 -> 1/3
    assert abs(compute_box_iou(b1, b3) - (50.0 / 150.0)) < 1e-6, "Half overlap IoU must be 1/3"

    b4 = [20.0, 20.0, 30.0, 30.0]
    assert compute_box_iou(b1, b4) == 0.0, "Disjoint IoU must be 0.0"
    print("✓ test_box_iou passed.")


def test_toy_case_known_answer():
    """
    Analytic Toy Case:
    1 Image, Class 0 has 2 Ground Truth instances:
      GT 1: Human [10, 10, 50, 50], Object [60, 60, 100, 100]
      GT 2: Human [100, 100, 150, 150], Object [160, 160, 200, 200]

    Predictions in order of score:
      Pred A: score 0.9, matches GT 1 (IoU=1.0) -> TP (1/1, rec 1/2 = 0.5)
      Pred B: score 0.8, matches GT 1 (IoU=1.0) -> FP (duplicate, 1/2, rec 0.5)
      Pred C: score 0.7, matches GT 2 (IoU=1.0) -> TP (2/3, rec 2/2 = 1.0)
      Pred D: score 0.6, disjoint (IoU=0.0)    -> FP (2/4, rec 1.0)

    P-R Envelope:
      Recall in [0, 0.5]: Prec = 1.0 -> Area = 0.5 * 1.0 = 0.5
      Recall in [0.5, 1.0]: Prec = 2/3 -> Area = 0.5 * (2/3) = 1/3
      Total Area (AP) = 0.5 + 1/3 = 5/6 = 0.8333333333333334
    """
    evaluator = HICODetBoxEvaluator(num_classes=2, iou_thresh=0.5)

    gt_objs = [
        {"id": 1, "bbox_human": [10, 50, 10, 50], "bbox_object": [60, 100, 60, 100]},
        {"id": 1, "bbox_human": [100, 150, 100, 150], "bbox_object": [160, 200, 160, 200]},
    ]
    evaluator.add_ground_truth(img_idx=0, gt_objects=gt_objs)

    preds = [
        ([10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0], 0, 0.9),    # Pred A -> TP
        ([10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0], 0, 0.8),    # Pred B -> FP (dup)
        ([100.0, 100.0, 150.0, 150.0], [160.0, 160.0, 200.0, 200.0], 0, 0.7), # Pred C -> TP
        ([0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0], 0, 0.6),              # Pred D -> FP
    ]
    evaluator.add_predictions(img_idx=0, triplets=preds)

    res = evaluator.evaluate_box_level()
    ap_c0 = res["per_class_ap"][0]
    expected_ap = 5.0 / 6.0
    print(f"Toy Case AP: {ap_c0:.8f}, Expected: {expected_ap:.8f}")
    assert abs(ap_c0 - expected_ap) < 1e-6, f"Mismatch in toy AP! Got {ap_c0}, expected {expected_ap}"
    print("✓ test_toy_case_known_answer passed.")


def test_asymmetric_iou():
    """
    Tests that min(IoU_h, IoU_o) >= 0.5 is enforced:
    If IoU_h >= 0.5 but IoU_o < 0.5, must be FP.
    If IoU_o >= 0.5 but IoU_h < 0.5, must be FP.
    """
    evaluator = HICODetBoxEvaluator(num_classes=1, iou_thresh=0.5)
    gt_objs = [
        {"id": 1, "bbox_human": [0, 10, 0, 10], "bbox_object": [20, 30, 20, 30]}
    ]
    evaluator.add_ground_truth(img_idx=0, gt_objects=gt_objs)

    # Pred 1: Human exact match (1.0), Object disjoint (0.0) -> FP
    # Pred 2: Object exact match (1.0), Human disjoint (0.0) -> FP
    preds = [
        ([0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0], 0, 0.9),
        ([100.0, 100.0, 110.0, 110.0], [20.0, 20.0, 30.0, 30.0], 0, 0.8),
    ]
    evaluator.add_predictions(img_idx=0, triplets=preds)

    res = evaluator.evaluate_box_level()
    assert res["per_class_ap"][0] == 0.0, "Asymmetric IoU must not trigger True Positive!"
    print("✓ test_asymmetric_iou passed.")


if __name__ == "__main__":
    print("Running HICO Evaluator Unit Tests...")
    test_box_iou()
    test_toy_case_known_answer()
    test_asymmetric_iou()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
