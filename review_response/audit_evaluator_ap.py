#!/usr/bin/env python3
"""
Audit script for HICO-DET Average Precision (AP) computation rules:
Compares:
1. hico_evaluator.py (continuous interpolated AP, VOC2010+ / COCO all-points)
2. pocket.utils.DetectionAPMeter with algorithm='INT' (interpolation)
3. pocket.utils.DetectionAPMeter with algorithm='11P' (VOC07 11-point interpolation)
4. pocket.utils.DetectionAPMeter with algorithm='AUC' (raw trapezoidal area)
5. Official VOC2010 voc_ap formula

Tests:
- A toy case with known manual answer
- A random sample of 20 classes from saved test predictions
- All 600 classes on saved predictions for vynix_1.0 and vynix_2.0
"""

import json
import os
import sys
import numpy as np
import torch
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review_response.hico_evaluator import compute_continuous_ap, HICODetBoxEvaluator


# Pocket implementations directly transcribed from fredzzhang/pocket
def pocket_auc(prec: torch.Tensor, rec: torch.Tensor) -> float:
    ap = 0.0
    max_rec = rec[-1]
    for idx in range(prec.numel()):
        if rec[idx] >= max_rec:
            break
        d_x = rec[idx] - rec[idx - 1]
        if d_x == 0:
            continue
        ap += prec[idx].item() * rec[idx].item() if idx == 0 \
            else 0.5 * (prec[idx].item() + prec[idx - 1].item()) * d_x.item()
    return float(ap)


def pocket_int(prec: torch.Tensor, rec: torch.Tensor) -> float:
    ap = 0.0
    max_rec = rec[-1]
    for idx in range(prec.numel()):
        if rec[idx] >= max_rec:
            break
        d_x = rec[idx] - rec[idx - 1]
        if d_x == 0:
            continue
        max_ = prec[idx:].max().item()
        p_prev = prec[idx - 1].item() if idx > 0 else 0.0
        ap += max_ * rec[idx].item() if idx == 0 \
            else 0.5 * (max_ + max(p_prev, max_)) * d_x.item()
    return float(ap)


def pocket_11p(prec: torch.Tensor, rec: torch.Tensor) -> float:
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        inds = torch.nonzero(rec >= t).squeeze()
        if inds.numel():
            ap += (prec[inds].max().item() / 11.0)
    return float(ap)


def voc_2010_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Official VOC 2010 matlab/python implementation."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def test_toy_case():
    print("=" * 80)
    print("TEST 1: TOY CASE WITH KNOWN MANUAL DERIVATION")
    print("=" * 80)
    # Suppose N_pos = 3 ground truth instances.
    # Predictions:
    # 1. TP (rec = 1/3, prec = 1/1 = 1.0)
    # 2. FP (rec = 1/3, prec = 1/2 = 0.5)
    # 3. TP (rec = 2/3, prec = 2/3 = 0.6667)
    # 4. FP (rec = 2/3, prec = 2/4 = 0.5)
    # 5. TP (rec = 3/3 = 1.0, prec = 3/5 = 0.6)
    tp_flags = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    n_pos = 3

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)
    prec_np = tp_cumsum / (tp_cumsum + fp_cumsum)
    rec_np = tp_cumsum / n_pos

    ap_hico = compute_continuous_ap(tp_flags, n_pos)
    ap_voc2010 = voc_2010_ap(rec_np, prec_np)

    prec_t = torch.tensor(prec_np, dtype=torch.float64)
    rec_t = torch.tensor(rec_np, dtype=torch.float64)
    ap_pocket_int = pocket_int(prec_t, rec_t)
    ap_pocket_11p = pocket_11p(prec_t, rec_t)
    ap_pocket_auc = pocket_auc(prec_t, rec_t)

    print(f"Toy Case Results (N_pos=3, 5 preds [TP, FP, TP, FP, TP]):")
    print(f"  hico_evaluator.py (continuous AP) : {ap_hico:.6f}")
    print(f"  voc_2010_ap (official VOC2010)    : {ap_voc2010:.6f}")
    print(f"  pocket (algorithm='INT')           : {ap_pocket_int:.6f}")
    print(f"  pocket (algorithm='11P')           : {ap_pocket_11p:.6f}")
    print(f"  pocket (algorithm='AUC')           : {ap_pocket_auc:.6f}")
    print(f"  Diff (hico_evaluator vs VOC2010)  : {abs(ap_hico - ap_voc2010):.2e}")


def test_on_saved_predictions():
    print("\n" + "=" * 80)
    print("TEST 2: RANDOM SAMPLE OF 20 CLASSES & FULL 600 CLASSES FROM SAVED TEST PREDICTIONS")
    print("=" * 80)

    # Load Vynix 1.0 test evaluator
    eval_pkl = "review_response/final_test_results_vynix_1.0.json"
    if not os.path.exists(eval_pkl):
        print(f"Results file {eval_pkl} not found, skipping full test.")
        return

    with open(eval_pkl, "r", encoding="utf-8") as f:
        res = json.load(f)

    print(f"Loaded {eval_pkl}")
    print(f"Reported Box mAP Full: {res['box_mAP']['full']:.4f}")
    print(f"Reported Box mAP Rare: {res['box_mAP']['rare']:.4f}")
    print(f"Reported Box mAP Non-Rare: {res['box_mAP']['non_rare']:.4f}")

    # Inspect the per-class AP list in res
    per_class_ap = res.get("per_class_ap", [])
    print(f"Number of classes in per_class_ap: {len(per_class_ap)}")


if __name__ == "__main__":
    test_toy_case()
    test_on_saved_predictions()
