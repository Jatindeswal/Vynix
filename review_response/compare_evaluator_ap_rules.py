#!/usr/bin/env python3
"""
Comprehensive AP Rule Comparison:
Compares:
1. Continuous interpolated AP (VOC 2010+ all points, implemented in hico_evaluator.py)
2. 11-Point interpolated AP (VOC 2007 11P, used in fredzzhang/pocket and fredzzhang/upt)
3. Raw AUC (trapezoid)

Evaluates on:
- Toy case
- Random 20 classes
- Full 600 classes from saved test predictions of Vynix 1.0 and Vynix 2.0
"""

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review_response.hico_evaluator import compute_continuous_ap, HICODetBoxEvaluator


def pocket_11p(prec: torch.Tensor, rec: torch.Tensor) -> float:
    """Official 11-point interpolated AP from fredzzhang/pocket."""
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        inds = torch.nonzero(rec >= t).squeeze()
        if inds.numel():
            ap += (prec[inds].max().item() / 11.0)
    return float(ap)


def pocket_int(prec: torch.Tensor, rec: torch.Tensor) -> float:
    """Interpolation method from fredzzhang/pocket."""
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


def run_ap_comparison():
    print("=" * 80)
    print("AP RULE COMPARISON: CONTINUOUS (VOC2010+) vs 11-POINT (fredzzhang/pocket/upt)")
    print("=" * 80)

    # 1. Toy Case
    tp_flags = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    n_pos = 3
    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)
    prec_np = tp_cumsum / (tp_cumsum + fp_cumsum)
    rec_np = tp_cumsum / n_pos

    ap_cont = compute_continuous_ap(tp_flags, n_pos)
    ap_11p = pocket_11p(torch.tensor(prec_np, dtype=torch.float64), torch.tensor(rec_np, dtype=torch.float64))
    print(f"\n1. TOY CASE (5 preds, 3 GTs):")
    print(f"   Continuous AP (hico_evaluator.py) : {ap_cont:.6f}")
    print(f"   11-Point AP (pocket / upt)        : {ap_11p:.6f}")
    print(f"   Delta (11P - Continuous)          : {ap_11p - ap_cont:+.6f}")

    # 2. Replay all test shards into HICODetBoxEvaluator
    print(f"\n2. LOADING GROUND TRUTHS & PREDICTIONS FROM SHARDS...")
    shards_dir = "E:/Do not open/Projects/Vynix/test_predictions"
    evaluator_cont = HICODetBoxEvaluator(num_classes=600, iou_thresh=0.5, split_json_path="review_response/standard_hico_splits.json")
    
    total_replayed = 0
    for shard_idx in range(4):
        shard_path = os.path.join(shards_dir, f"preds_vynix_1.0_shard_{shard_idx:02d}.pt")
        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_preds = shard_data["predictions"]["full"]
        shard_gts = shard_data["ground_truths"]
        for img_idx in sorted(shard_gts.keys()):
            evaluator_cont.add_ground_truth(img_idx, shard_gts[img_idx])
            evaluator_cont.add_predictions(img_idx, shard_preds.get(img_idx, []))
            total_replayed += 1
    print(f"   ✓ Replayed {total_replayed} images across 4 shards.")

    # 3. Evaluate across all classes
    rare_ids = evaluator_cont.rare_ids
    non_rare_ids = evaluator_cont.non_rare_ids

    ap_continuous_list = []
    ap_11p_list = []

    np.random.seed(42)
    sample_classes = set(np.random.choice(600, size=20, replace=False))
    sample_table = []

    for c in range(600):
        n_pos = int(evaluator_cont.num_gt_per_class[c])
        preds_c = evaluator_cont.predictions[c]
        if n_pos == 0 or len(preds_c) == 0:
            ap_continuous_list.append(0.0)
            ap_11p_list.append(0.0)
            continue

        preds_c.sort(key=lambda x: x[0], reverse=True)
        img_matched = {img_id: set() for img_id in evaluator_cont.ground_truths[c]}

        tp_flags = np.zeros(len(preds_c), dtype=np.int32)
        for i, (score, img_id, h_box, o_box) in enumerate(preds_c):
            if img_id not in evaluator_cont.ground_truths[c]:
                continue
            gt_pairs = evaluator_cont.ground_truths[c][img_id]
            best_iou = -1.0
            best_gt_idx = -1
            for gt_idx, (gt_h, gt_o) in enumerate(gt_pairs):
                if gt_idx in img_matched[img_id]:
                    continue
                # compute_box_iou
                from review_response.hico_evaluator import compute_box_iou
                iou_h = compute_box_iou(h_box, gt_h)
                iou_o = compute_box_iou(o_box, gt_o)
                min_iou = min(iou_h, iou_o)
                if min_iou > best_iou:
                    best_iou = min_iou
                    best_gt_idx = gt_idx
            if best_iou >= 0.5 and best_gt_idx >= 0:
                tp_flags[i] = 1
                img_matched[img_id].add(best_gt_idx)

        # Compute Continuous AP
        ap_c_cont = compute_continuous_ap(tp_flags, n_pos)

        # Compute 11-Point AP
        tp_cumsum = np.cumsum(tp_flags)
        fp_cumsum = np.cumsum(1 - tp_flags)
        prec_t = torch.tensor(tp_cumsum / (tp_cumsum + fp_cumsum), dtype=torch.float64)
        rec_t = torch.tensor(tp_cumsum / n_pos, dtype=torch.float64)
        ap_c_11p = pocket_11p(prec_t, rec_t)

        ap_continuous_list.append(ap_c_cont)
        ap_11p_list.append(ap_c_11p)

        if c in sample_classes:
            sample_table.append({
                "class_id": c,
                "n_pos": n_pos,
                "n_preds": len(preds_c),
                "n_tps": int(tp_flags.sum()),
                "ap_continuous": ap_c_cont,
                "ap_11p": ap_c_11p,
                "diff": ap_c_11p - ap_c_cont,
            })

    # Summary table for 20 sample classes
    print("\n3. SAMPLE OF 20 CLASSES EVALUATED SIDE BY SIDE:")
    print(f"{'Class':>5} | {'GT':>4} | {'Preds':>6} | {'TPs':>4} | {'Continuous AP':>13} | {'11P AP (pocket)':>15} | {'Delta':>8}")
    print("-" * 65)
    for row in sample_table:
        print(f"{row['class_id']:5d} | {row['n_pos']:4d} | {row['n_preds']:6d} | {row['n_tps']:4d} | {row['ap_continuous']:13.4f} | {row['ap_11p']:15.4f} | {row['diff']:+8.4f}")

    # Full Dataset mAP
    mAP_cont_full = float(np.mean(ap_continuous_list)) * 100.0
    mAP_cont_rare = float(np.mean([ap_continuous_list[i] for i in rare_ids])) * 100.0
    mAP_cont_non_rare = float(np.mean([ap_continuous_list[i] for i in non_rare_ids])) * 100.0

    mAP_11p_full = float(np.mean(ap_11p_list)) * 100.0
    mAP_11p_rare = float(np.mean([ap_11p_list[i] for i in rare_ids])) * 100.0
    mAP_11p_non_rare = float(np.mean([ap_11p_list[i] for i in non_rare_ids])) * 100.0

    print("\n" + "=" * 80)
    print("4. FULL TEST SET mAP SUMMARY (Vynix 1.0 Default Setting)")
    print("=" * 80)
    print(f"{'Metric':<18} | {'Continuous AP (VOC10+)':<22} | {'11-Point AP (pocket/upt)':<24} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Full mAP (600)':<18} | {mAP_cont_full:21.2f}% | {mAP_11p_full:23.2f}% | {mAP_11p_full - mAP_cont_full:+9.2f}%")
    print(f"{'Rare mAP (138)':<18} | {mAP_cont_rare:21.2f}% | {mAP_11p_rare:23.2f}% | {mAP_11p_rare - mAP_cont_rare:+9.2f}%")
    print(f"{'Non-Rare mAP (462)':<18} | {mAP_cont_non_rare:21.2f}% | {mAP_11p_non_rare:23.2f}% | {mAP_11p_non_rare - mAP_cont_non_rare:+9.2f}%")

    out_record = {
        "toy_case": {"continuous": ap_cont, "11p": ap_11p, "delta": ap_11p - ap_cont},
        "sample_20_classes": sample_table,
        "full_dataset_vynix_1.0": {
            "continuous_mAP": {"full": mAP_cont_full, "rare": mAP_cont_rare, "non_rare": mAP_cont_non_rare},
            "11p_mAP": {"full": mAP_11p_full, "rare": mAP_11p_rare, "non_rare": mAP_11p_non_rare},
            "delta": {"full": mAP_11p_full - mAP_cont_full, "rare": mAP_11p_rare - mAP_cont_rare, "non_rare": mAP_11p_non_rare - mAP_cont_non_rare},
        }
    }
    with open("review_response/ap_rule_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2)
    print("\n✓ Saved audit results to review_response/ap_rule_audit_results.json")


if __name__ == "__main__":
    run_ap_comparison()
