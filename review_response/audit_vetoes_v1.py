#!/usr/bin/env python3
"""
Recomputes exact GT-matched and unmatched veto/rescue counts for Vynix 1.0
from the saved test predictions and test parquet annotations.
"""

import ast
import glob
import json
import math
import os
import sys
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import (
    HOIMeta,
    compute_box_distance,
    compute_iou,
    compute_soft_geometric_gate,
)


def parse_raw_objects(raw_objs):
    if isinstance(raw_objs, str):
        return ast.literal_eval(raw_objs) if raw_objs.strip() else []
    return raw_objs if isinstance(raw_objs, list) else []


def main():
    meta = HOIMeta("E:/Dataset")
    preds_path = r"E:\Do not open\Projects\Vynix\test_predictions\test_predictions_vynix_1.0_full.pt"
    print(f"Loading predictions from {preds_path}...")
    saved_preds = torch.load(preds_path, map_location="cpu", weights_only=False)

    # We use 'no_geom_gate' or 'full' predictions to extract candidate pairs
    preds_full = saved_preds["full"]
    print(f"Loaded predictions for {len(preds_full)} images.")

    test_files = sorted(glob.glob(r"E:\Dataset\data\test-*.parquet"))

    veto_stats = {
        "total_hypotheses": 0,
        "contact_hypotheses": 0,
        "non_contact_hypotheses": 0,
        "soft_vetoes_frozen": 0,
        "soft_vetoes_frozen_box_matched": 0,
        "soft_vetoes_frozen_img_matched": 0,
        "soft_vetoes_frozen_unmatched": 0,
        "soft_vetoes_sigma08": 0,
        "soft_vetoes_sigma08_box_matched": 0,
        "soft_vetoes_sigma08_img_matched": 0,
        "soft_vetoes_sigma08_unmatched": 0,
        "hard_vetoes": 0,
        "hard_vetoes_box_matched": 0,
        "hard_vetoes_img_matched": 0,
        "hard_vetoes_unmatched": 0,
        "rescued_frozen_box_matched": 0,
        "rescued_frozen_img_matched": 0,
        "rescued_sigma08_box_matched": 0,
        "rescued_sigma08_img_matched": 0,
    }

    processed_count = 0
    with tqdm(total=len(preds_full), desc="Auditing Vetoes (v1.0)") as pbar:
        for f in test_files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    eval_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    raw_objs = parse_raw_objects(row.get("objects", []))
                    gt_box_list = []
                    gt_img_hois = set()
                    for item in raw_objs:
                        h_id = item.get("id")
                        if h_id is not None:
                            c = int(h_id) - 1
                            if 0 <= c < meta.num_classes:
                                gt_img_hois.add(c)
                                hb = item.get("bbox_human")
                                ob = item.get("bbox_object")
                                if hb is not None and ob is not None and len(hb) >= 4 and len(ob) >= 4:
                                    h_box = [float(hb[0]), float(hb[2]), float(hb[1]), float(hb[3])]
                                    o_box = [float(ob[0]), float(ob[2]), float(ob[1]), float(ob[3])]
                                    gt_box_list.append((h_box, o_box, c))

                    # Parse image dimensions for box distance
                    img_data = row["image"]
                    img_w, img_h = 640, 640  # fallback
                    if isinstance(img_data, dict) and img_data.get("bytes") is not None:
                        from PIL import Image
                        import io
                        with Image.open(io.BytesIO(img_data["bytes"])) as im:
                            img_w, img_h = im.size

                    triplets = preds_full.get(eval_idx, [])
                    for (p_box, o_box, hoi_id, _) in triplets:
                        verb = meta.hoi_to_verb[hoi_id]
                        iou_val = compute_iou(p_box, o_box)
                        b_dist = compute_box_distance(p_box, o_box, img_w, img_h)

                        gate_frozen = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=0.01)
                        gate_sigma08 = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=0.08)

                        is_contact_verb = (compute_soft_geometric_gate(verb, 0.0, 0.1, sigma_contact=0.08) < 1.0)

                        veto_stats["total_hypotheses"] += 1
                        if is_contact_verb:
                            veto_stats["contact_hypotheses"] += 1
                        else:
                            veto_stats["non_contact_hypotheses"] += 1

                        box_matched = False
                        for (gt_h, gt_o, gt_hid) in gt_box_list:
                            if gt_hid == hoi_id and compute_iou(p_box, gt_h) >= 0.5 and compute_iou(o_box, gt_o) >= 0.5:
                                box_matched = True
                                break
                        img_matched = (hoi_id in gt_img_hois)

                        # Frozen soft veto (sigma=0.01, gate < 0.05)
                        if gate_frozen < 0.05:
                            veto_stats["soft_vetoes_frozen"] += 1
                            if box_matched:
                                veto_stats["soft_vetoes_frozen_box_matched"] += 1
                            if img_matched:
                                veto_stats["soft_vetoes_frozen_img_matched"] += 1
                            if not img_matched:
                                veto_stats["soft_vetoes_frozen_unmatched"] += 1

                        # Nominal sigma=0.08 soft veto (< 0.05)
                        if gate_sigma08 < 0.05:
                            veto_stats["soft_vetoes_sigma08"] += 1
                            if box_matched:
                                veto_stats["soft_vetoes_sigma08_box_matched"] += 1
                            if img_matched:
                                veto_stats["soft_vetoes_sigma08_img_matched"] += 1
                            if not img_matched:
                                veto_stats["soft_vetoes_sigma08_unmatched"] += 1

                        # Hard veto (iou == 0 for contact verb)
                        if is_contact_verb and iou_val == 0.0:
                            veto_stats["hard_vetoes"] += 1
                            if box_matched:
                                veto_stats["hard_vetoes_box_matched"] += 1
                            if img_matched:
                                veto_stats["hard_vetoes_img_matched"] += 1
                            if not img_matched:
                                veto_stats["hard_vetoes_unmatched"] += 1

                        # Rescued hypotheses (contact verb, IoU==0, d_box <= sigma, and GT-matched)
                        if is_contact_verb and iou_val == 0.0:
                            if b_dist <= 0.01:
                                if box_matched:
                                    veto_stats["rescued_frozen_box_matched"] += 1
                                if img_matched:
                                    veto_stats["rescued_frozen_img_matched"] += 1
                            if b_dist <= 0.08:
                                if box_matched:
                                    veto_stats["rescued_sigma08_box_matched"] += 1
                                if img_matched:
                                    veto_stats["rescued_sigma08_img_matched"] += 1

    print("\n--- Updated Veto and Rescue Audit (Vynix 1.0) ---")
    for k, v in veto_stats.items():
        print(f"  {k:<35}: {v:,}")

    # Update final_test_results_vynix_1.0.json
    res_path = "review_response/final_test_results_vynix_1.0.json"
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["veto_and_rescue_audit"] = veto_stats
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Updated {res_path} with split veto/rescue counts.")


if __name__ == "__main__":
    main()
