#!/usr/bin/env python3
"""
Task 4: Standalone Veto & Rescue Count Auditor (HICO-DET Test Set, 9,658 images)
Recomputes total candidate (pair, verb) hypotheses, soft-gate vetoes (gate < 0.05),
hard-gate vetoes (IoU == 0), and rescued border-touching hypotheses, split by
whether hoi_id matches the image's ground-truth positive_objects.
"""
import argparse
import csv
import glob
import io
import math
import os
import sys

import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import (
    COCO_CLASSES,
    CONTACT_VERBS,
    HOIMeta,
    compute_box_distance,
    compute_iou,
    normalize_name,
    parse_int_list,
)


def audit_detector(detector_model: str, dataset_dir: str, device: str, sigma: float = 0.08, limit: int = None):
    meta = HOIMeta(dataset_dir)
    yolo = YOLO(detector_model)
    test_files = sorted(glob.glob(os.path.join(dataset_dir, "data", "test-*.parquet")))

    stats = {
        "detector": detector_model,
        "images": 0,
        "total_pairs_top50": 0,
        "total_hypotheses": 0,
        "hard_veto_total": 0,
        "hard_veto_gt_match": 0,
        "hard_veto_unmatched": 0,
        "soft_code_veto_total": 0,
        "soft_code_veto_gt_match": 0,
        "soft_code_veto_unmatched": 0,
        "soft_code_rescued_total": 0,
        "soft_code_rescued_gt_match": 0,
        "soft_code_rescued_unmatched": 0,
        "soft_paper_veto_total": 0,
        "soft_paper_veto_gt_match": 0,
        "soft_paper_veto_unmatched": 0,
    }

    for f in test_files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=32):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                if limit and stats["images"] >= limit:
                    return stats
                stats["images"] += 1
                gt_set = set(parse_int_list(str(row.get("positive_objects", ""))))

                img_data = row["image"]
                if isinstance(img_data, dict) and img_data.get("bytes") is not None:
                    pil_img = Image.open(io.BytesIO(img_data["bytes"]))
                elif isinstance(img_data, Image.Image):
                    pil_img = img_data
                else:
                    continue
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                results = yolo(pil_img, device=device, verbose=False)
                persons, objects = [], []
                for r in results:
                    for i in range(len(r.boxes)):
                        conf = float(r.boxes.conf[i].item())
                        if conf < 0.08:
                            continue
                        cls_id = int(r.boxes.cls[i].item())
                        box = r.boxes.xyxy[i].tolist()
                        if cls_id == 0:
                            persons.append((box, conf))
                        else:
                            objects.append((box, conf, cls_id))

                if not persons or not objects:
                    continue

                img_w, img_h = pil_img.size
                pairs = [(p, o) for p in persons for o in objects]
                if len(pairs) > 50:
                    pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
                    pairs = pairs[:50]

                for (p_box, _), (o_box, _, o_cls) in pairs:
                    coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
                    if not coco_name:
                        continue
                    entries = meta.obj_to_entries.get(normalize_name(coco_name), [])
                    if not entries:
                        continue

                    stats["total_pairs_top50"] += 1
                    iou_val = compute_iou(p_box, o_box)
                    b_dist = compute_box_distance(p_box, o_box, img_w, img_h)

                    for verb, _, hoi_id in entries:
                        stats["total_hypotheses"] += 1
                        is_gt = hoi_id in gt_set
                        is_contact_zero_iou = (verb in CONTACT_VERBS) and (iou_val == 0.0)

                        if is_contact_zero_iou:
                            stats["hard_veto_total"] += 1
                            stats["hard_veto_gt_match" if is_gt else "hard_veto_unmatched"] += 1

                            g_code = math.exp(-((b_dist / sigma) ** 2))
                            g_paper = math.exp(-(b_dist ** 2) / (2.0 * (sigma ** 2)))

                            if g_code < 0.05:
                                stats["soft_code_veto_total"] += 1
                                stats["soft_code_veto_gt_match" if is_gt else "soft_code_veto_unmatched"] += 1
                            else:
                                stats["soft_code_rescued_total"] += 1
                                stats["soft_code_rescued_gt_match" if is_gt else "soft_code_rescued_unmatched"] += 1

                            if g_paper < 0.05:
                                stats["soft_paper_veto_total"] += 1
                                stats["soft_paper_veto_gt_match" if is_gt else "soft_paper_veto_unmatched"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-csv", default="review_response/04_veto_audit_results.csv")
    args = parser.parse_args()

    rows = [
        audit_detector("yolov8m.pt", args.dataset_dir, args.device, limit=args.limit),
        audit_detector("yolov8x.pt", args.dataset_dir, args.device, limit=args.limit),
    ]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
