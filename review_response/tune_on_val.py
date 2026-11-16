#!/usr/bin/env python3
"""
Hyperparameter Tuning on Train-Derived Validation Split (10% = 3,812 images)
============================================================================
Clean Validation Protocol for Vynix 1.0 (YOLOv8m) and Vynix 2.0 (YOLOv8x).
Trained strictly on train_90_ids.json ONLY.

Features:
- Extracts val_10 candidate pairs and CLIP visual features at detector_conf=0.05
  so detector_threshold in {0.05, 0.08, 0.12} can be evaluated in-memory.
- Evaluates Box-Level Default mAP (Full, Rare 138, Non-Rare 462).
- Coordinates-wise sweeps:
    * Vynix 2.0:
        1. sigma in {0.04, 0.06, 0.08, 0.10, 0.12, 0.16} (baseline: lambda=0.90, gamma_obj=0.10, det_thresh=0.08)
        2. lambda in {0.70, 0.80, 0.90, 1.00} (with optimal sigma)
        3. gamma_obj in {0.00, 0.05, 0.10, 0.20} (with optimal sigma, lambda)
        4. detector_thresh in {0.05, 0.08, 0.12} (with optimal sigma, lambda, gamma_obj)
    * Vynix 1.0:
        1. sigma in {0.04, 0.06, 0.08, 0.10, 0.12, 0.16} (det_thresh=0.08, lambda=1.0, gamma_obj=0.0)
        2. detector_thresh in {0.05, 0.08, 0.12} (with optimal sigma, lambda=1.0, gamma_obj=0.0)
- Saves full evaluation trajectory to review_response/val_tuning_results.json.
- Freezes optimal configurations to review_response/frozen_hyperparameters.json.
"""

import argparse
import ast
import glob
import io
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import (
    COCO_CLASSES,
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    compute_box_distance,
    compute_iou,
    compute_soft_geometric_gate,
    compute_spatial_vector,
    compute_union_box,
    normalize_name,
)
from review_response.hico_evaluator import HICODetBoxEvaluator


def extract_val_cache(
    dataset_dir: str,
    model_dir: str,
    val_ids_path: str,
    output_cache: str,
    detector_model: str = "yolov8x.pt",
    clip_model: str = "openai/clip-vit-base-patch16",
    device: str = "cuda",
    detector_conf: float = 0.05,
    limit: Optional[int] = None,
):
    print("=" * 80)
    print(f"  EXTRACTING VAL CACHE -> {output_cache}")
    print(f"  Detector: {detector_model} (conf={detector_conf}) | Adapter: {model_dir}")
    print("=" * 80)

    meta = HOIMeta(dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=device,
        detector_conf=detector_conf,
        use_3stream=True,
        detector_model=detector_model,
        clip_model=clip_model,
    )
    extractor.yolo.to(device).eval()

    text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(device)
    obj_text_weights = extractor.build_object_text_weights(meta)

    cache_path = os.path.join(model_dir, "vynix_full_cache.pt")
    model_path = os.path.join(model_dir, "vynix_full_model.pth")
    cache_data = torch.load(cache_path, map_location=device, weights_only=True)
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_data["cache_keys"],
        cache_values=cache_data["cache_values"],
        use_spatial_mlp=True,
        num_classes=meta.num_classes,
    ).to(device).eval()
    if os.path.exists(model_path):
        adapter.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    with open(val_ids_path, "r", encoding="utf-8") as f:
        val_indices = set(json.load(f))
    print(f"  Loaded {len(val_indices)} validation image IDs.")

    train_files = sorted(glob.glob(os.path.join(dataset_dir, "data", "train-*.parquet")))
    n_target = min(len(val_indices), limit) if limit else len(val_indices)

    cached_records = []
    global_img_idx = 0
    processed_count = 0

    t0 = time.time()
    with tqdm(total=n_target, desc=f"  Val Cache ({os.path.basename(detector_model)})", unit="img") as pbar:
        for shard_idx, f in enumerate(train_files):
            shard_cache_path = f"{output_cache}.shard_{shard_idx:02d}.pt"
            if os.path.exists(shard_cache_path):
                shard_recs = torch.load(shard_cache_path, map_location="cpu", weights_only=False)
                cached_records.extend(shard_recs)
                pbar.update(len(shard_recs))
                processed_count += len(shard_recs)
                pf = pq.ParquetFile(f)
                global_img_idx += pf.metadata.num_rows
                print(f"\n  [Resume] Shard {shard_idx:02d} loaded from disk ({len(shard_recs)} records).")
                continue

            shard_records = []
            pf = pq.ParquetFile(f)
            reached_limit = False
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    curr_idx = global_img_idx
                    global_img_idx += 1

                    if curr_idx not in val_indices:
                        continue
                    if limit and processed_count >= limit:
                        reached_limit = True
                        break

                    eval_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    raw_objs = row.get("objects", [])
                    if isinstance(raw_objs, str):
                        raw_objs = ast.literal_eval(raw_objs) if raw_objs.strip() else []

                    img_data = row["image"]
                    if isinstance(img_data, dict) and img_data.get("bytes") is not None:
                        pil_img = Image.open(io.BytesIO(img_data["bytes"]))
                    elif isinstance(img_data, Image.Image):
                        pil_img = img_data
                    else:
                        continue
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    persons, objects = extractor.detect(pil_img)
                    if not persons or not objects:
                        rec = {
                            "eval_idx": eval_idx,
                            "gt_objects": raw_objs,
                            "pairs_data": [],
                        }
                        cached_records.append(rec)
                        shard_records.append(rec)
                        del pil_img
                        continue

                    img_w, img_h = pil_img.size
                    pairs = [(p, o) for p in persons for o in objects]
                    if len(pairs) > 100:
                        pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
                        pairs = pairs[:100]

                    valid_pairs = []
                    for (p_box, p_conf), (o_box, o_conf, o_cls) in pairs:
                        coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
                        if not coco_name:
                            continue
                        hico_name = normalize_name(coco_name)
                        entries = meta.obj_to_entries.get(hico_name, [])
                        if not entries:
                            continue

                        u_box = compute_union_box(p_box, o_box)
                        s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)
                        iou_val = compute_iou(p_box, o_box)
                        b_dist = compute_box_distance(p_box, o_box, img_w, img_h)
                        valid_pairs.append({
                            "p_conf": p_conf,
                            "o_conf": o_conf,
                            "p_box": p_box,
                            "o_box": o_box,
                            "u_box": u_box,
                            "s_vec": s_vec,
                            "iou_val": iou_val,
                            "b_dist": b_dist,
                            "entries": entries,
                        })

                    if not valid_pairs:
                        rec = {
                            "eval_idx": eval_idx,
                            "gt_objects": raw_objs,
                            "pairs_data": [],
                        }
                        cached_records.append(rec)
                        shard_records.append(rec)
                        del pil_img
                        continue

                    boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)
                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch

                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights.T
                        final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
                        base_probs = torch.softmax(final_logits, dim=1).cpu()

                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

                    pairs_data = []
                    for k_idx, v in enumerate(valid_pairs):
                        cos_sim = 1.0
                        if v["entries"]:
                            hico_name = meta.hoi_to_obj[v["entries"][0][2]]
                            if hico_name in obj_text_weights:
                                t_obj = obj_text_weights[hico_name].cpu()
                                cos_sim = max(0.0, (f_objs_norm[k_idx] @ t_obj).item())

                        pairs_data.append({
                            "p_conf": v["p_conf"],
                            "o_conf": v["o_conf"],
                            "p_box": v["p_box"],
                            "o_box": v["o_box"],
                            "iou_val": v["iou_val"],
                            "b_dist": v["b_dist"],
                            "entries": v["entries"],
                            "base_probs": base_probs[k_idx],
                            "cos_sim": cos_sim,
                        })

                    rec = {
                        "eval_idx": eval_idx,
                        "gt_objects": raw_objs,
                        "pairs_data": pairs_data,
                    }
                    cached_records.append(rec)
                    shard_records.append(rec)
                    del pil_img

                if reached_limit:
                    break

            if shard_records:
                torch.save(shard_records, shard_cache_path)
                print(f"\n  ✓ Saved Shard {shard_idx:02d}: {len(shard_records)} records to {shard_cache_path}")

            if reached_limit:
                break

    dt = time.time() - t0
    print(f"\n  [Extraction Complete] {len(cached_records)} records in {dt/60:.2f} min. Saving to {output_cache}...")
    torch.save(cached_records, output_cache)
    print("  ✓ Saved successfully.")
    return cached_records


def evaluate_grid_point(
    cached_records: List[Dict],
    meta: HOIMeta,
    det_lambda: float,
    obj_gamma: float,
    sigma_contact: float,
    det_thresh: float = 0.08,
    split_json_path: str = "review_response/standard_hico_splits.json",
) -> Dict[str, float]:
    evaluator = HICODetBoxEvaluator(
        num_classes=meta.num_classes,
        iou_thresh=0.5,
        split_json_path=split_json_path,
    )

    for rec in cached_records:
        eval_idx = rec["eval_idx"]
        evaluator.add_ground_truth(eval_idx, rec["gt_objects"])

        # Filter pairs by detector threshold
        candidate_pairs = [
            p for p in rec["pairs_data"]
            if p["p_conf"] >= det_thresh and p["o_conf"] >= det_thresh
        ]
        if len(candidate_pairs) > 50:
            candidate_pairs.sort(key=lambda x: x["p_conf"] * x["o_conf"], reverse=True)
            candidate_pairs = candidate_pairs[:50]

        triplets = []
        for p in candidate_pairs:
            p_conf, o_conf = p["p_conf"], p["o_conf"]
            p_box, o_box = p["p_box"], p["o_box"]
            iou_val, b_dist = p["iou_val"], p["b_dist"]
            det_score = (p_conf * o_conf) ** det_lambda
            obj_gate = (p["cos_sim"] ** obj_gamma) if obj_gamma > 0 else 1.0

            for verb, _, hoi_id in p["entries"]:
                raw_prob = p["base_probs"][hoi_id].item()
                gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma_contact)
                conf = det_score * raw_prob * gate * obj_gate
                triplets.append((p_box, o_box, hoi_id, conf))

        evaluator.add_predictions(eval_idx, triplets)

    return evaluator.evaluate_box_level(only_valid_gt=True)


def select_best_param(candidates: List[Tuple[float, Dict]], default_val: float) -> Tuple[float, float]:
    """
    Selection rule, fixed in advance:
    Pick the best-Full-mAP value; if within 0.1 of another, keep the one closest to current default.
    candidates: list of (param_val, metrics_dict)
    """
    max_map = max(m["mAP_full"] for _, m in candidates)
    close_candidates = [(v, m) for v, m in candidates if m["mAP_full"] >= max_map - 0.10]
    close_candidates.sort(key=lambda x: (round(abs(x[0] - default_val), 4), -x[1]["mAP_full"]))
    best_val, best_metrics = close_candidates[0]
    return best_val, best_metrics["mAP_full"]


def tune_vynix2(cached_records: List[Dict], meta: HOIMeta) -> Tuple[Dict, List[Dict]]:
    print("\n" + "=" * 80)
    print("  COORDINATE-WISE HYPERPARAMETER TUNING — VYNIX 2.0 (YOLOv8x)")
    print("=" * 80)

    history = []
    current_params = {
        "sigma": 0.08,
        "lambda": 0.90,
        "gamma_obj": 0.10,
        "detector_threshold": 0.08,
    }

    # Baseline coverage check
    base_res = evaluate_grid_point(
        cached_records, meta,
        det_lambda=current_params["lambda"],
        obj_gamma=current_params["gamma_obj"],
        sigma_contact=current_params["sigma"],
        det_thresh=current_params["detector_threshold"],
    )
    n_full = base_res["num_classes_full"]
    n_rare = base_res["num_classes_rare"]
    n_non_rare = base_res["num_classes_non_rare"]
    print(f"\n  [Validation Coverage] Classes with >=1 GT in val_10:")
    print(f"    Full: {n_full}/600 | Rare 138: {n_rare}/138 | Non-Rare 462: {n_non_rare}/462")
    print(f"    (Selection objective: Full mAP over these {n_full} classes. Tie-break: within 0.1 of peak closest to default)\n")

    # Step 1: Sweep sigma
    print("[Step 1] Sweeping sigma in {0.04, 0.06, 0.08, 0.10, 0.12, 0.16} (default: 0.08)...")
    sigma_grid = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16]
    candidates = []
    for s in sigma_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=current_params["lambda"],
            obj_gamma=current_params["gamma_obj"],
            sigma_contact=s,
            det_thresh=current_params["detector_threshold"],
        )
        print(f"  sigma={s:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_2.0",
            "step": "sigma_sweep",
            "sigma": s,
            "lambda": current_params["lambda"],
            "gamma_obj": current_params["gamma_obj"],
            "detector_threshold": current_params["detector_threshold"],
            "metrics": res,
        })
        candidates.append((s, res))
    best_sigma, best_map = select_best_param(candidates, default_val=0.08)
    current_params["sigma"] = best_sigma
    print(f"  => Selected sigma: {best_sigma} (mAP: {best_map:.2f}%)\n")

    # Step 2: Sweep lambda
    print(f"[Step 2] Sweeping lambda in {{0.70, 0.80, 0.90, 1.00}} with sigma={current_params['sigma']} (default: 0.90)...")
    lam_grid = [0.70, 0.80, 0.90, 1.00]
    candidates = []
    for l in lam_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=l,
            obj_gamma=current_params["gamma_obj"],
            sigma_contact=current_params["sigma"],
            det_thresh=current_params["detector_threshold"],
        )
        print(f"  lambda={l:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_2.0",
            "step": "lambda_sweep",
            "sigma": current_params["sigma"],
            "lambda": l,
            "gamma_obj": current_params["gamma_obj"],
            "detector_threshold": current_params["detector_threshold"],
            "metrics": res,
        })
        candidates.append((l, res))
    best_lam, best_map = select_best_param(candidates, default_val=0.90)
    current_params["lambda"] = best_lam
    print(f"  => Selected lambda: {best_lam} (mAP: {best_map:.2f}%)\n")

    # Step 3: Sweep gamma_obj
    print(f"[Step 3] Sweeping gamma_obj in {{0.00, 0.05, 0.10, 0.20}} with lambda={current_params['lambda']} (default: 0.10)...")
    gamma_grid = [0.00, 0.05, 0.10, 0.20]
    candidates = []
    for g in gamma_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=current_params["lambda"],
            obj_gamma=g,
            sigma_contact=current_params["sigma"],
            det_thresh=current_params["detector_threshold"],
        )
        print(f"  gamma_obj={g:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_2.0",
            "step": "gamma_obj_sweep",
            "sigma": current_params["sigma"],
            "lambda": current_params["lambda"],
            "gamma_obj": g,
            "detector_threshold": current_params["detector_threshold"],
            "metrics": res,
        })
        candidates.append((g, res))
    best_gamma, best_map = select_best_param(candidates, default_val=0.10)
    current_params["gamma_obj"] = best_gamma
    print(f"  => Selected gamma_obj: {best_gamma} (mAP: {best_map:.2f}%)\n")

    # Step 4: Sweep detector threshold
    print(f"[Step 4] Sweeping detector threshold in {{0.05, 0.08, 0.12}} (default: 0.08)...")
    thresh_grid = [0.05, 0.08, 0.12]
    candidates = []
    for t in thresh_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=current_params["lambda"],
            obj_gamma=current_params["gamma_obj"],
            sigma_contact=current_params["sigma"],
            det_thresh=t,
        )
        print(f"  det_thresh={t:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_2.0",
            "step": "det_thresh_sweep",
            "sigma": current_params["sigma"],
            "lambda": current_params["lambda"],
            "gamma_obj": current_params["gamma_obj"],
            "detector_threshold": t,
            "metrics": res,
        })
        candidates.append((t, res))
    best_thresh, best_map = select_best_param(candidates, default_val=0.08)
    current_params["detector_threshold"] = best_thresh
    print(f"  => Selected detector_threshold: {best_thresh} (mAP: {best_map:.2f}%)\n")

    return current_params, history


def tune_vynix1(cached_records: List[Dict], meta: HOIMeta) -> Tuple[Dict, List[Dict]]:
    print("\n" + "=" * 80)
    print("  COORDINATE-WISE HYPERPARAMETER TUNING — VYNIX 1.0 (YOLOv8m)")
    print("=" * 80)

    history = []
    current_params = {
        "sigma": 0.08,
        "lambda": 1.00,
        "gamma_obj": 0.00,
        "detector_threshold": 0.08,
    }

    # Baseline coverage check
    base_res = evaluate_grid_point(
        cached_records, meta,
        det_lambda=1.00,
        obj_gamma=0.00,
        sigma_contact=current_params["sigma"],
        det_thresh=current_params["detector_threshold"],
    )
    n_full = base_res["num_classes_full"]
    n_rare = base_res["num_classes_rare"]
    n_non_rare = base_res["num_classes_non_rare"]
    print(f"\n  [Validation Coverage] Classes with >=1 GT in val_10:")
    print(f"    Full: {n_full}/600 | Rare 138: {n_rare}/138 | Non-Rare 462: {n_non_rare}/462")
    print(f"    (Selection objective: Full mAP over these {n_full} classes. Tie-break: within 0.1 of peak closest to default)\n")

    # Step 1: Sweep sigma
    print("[Step 1] Sweeping sigma in {0.04, 0.06, 0.08, 0.10, 0.12, 0.16} (default: 0.08)...")
    sigma_grid = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16]
    candidates = []
    for s in sigma_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=1.00,
            obj_gamma=0.00,
            sigma_contact=s,
            det_thresh=current_params["detector_threshold"],
        )
        print(f"  sigma={s:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_1.0",
            "step": "sigma_sweep",
            "sigma": s,
            "lambda": 1.00,
            "gamma_obj": 0.00,
            "detector_threshold": current_params["detector_threshold"],
            "metrics": res,
        })
        candidates.append((s, res))
    best_sigma, best_map = select_best_param(candidates, default_val=0.08)
    current_params["sigma"] = best_sigma
    print(f"  => Selected sigma: {best_sigma} (mAP: {best_map:.2f}%)\n")

    # Step 2: Sweep detector threshold
    print(f"[Step 2] Sweeping detector threshold in {{0.05, 0.08, 0.12}} with sigma={current_params['sigma']} (default: 0.08)...")
    thresh_grid = [0.05, 0.08, 0.12]
    candidates = []
    for t in thresh_grid:
        res = evaluate_grid_point(
            cached_records, meta,
            det_lambda=1.00,
            obj_gamma=0.00,
            sigma_contact=current_params["sigma"],
            det_thresh=t,
        )
        print(f"  det_thresh={t:4.2f} -> Full ({n_full} cls): {res['mAP_full']:.2f}%, Rare ({n_rare} cls): {res['mAP_rare']:.2f}%, Non-Rare ({n_non_rare} cls): {res['mAP_non_rare']:.2f}%")
        history.append({
            "model": "vynix_1.0",
            "step": "det_thresh_sweep",
            "sigma": current_params["sigma"],
            "lambda": 1.00,
            "gamma_obj": 0.00,
            "detector_threshold": t,
            "metrics": res,
        })
        candidates.append((t, res))
    best_thresh, best_map = select_best_param(candidates, default_val=0.08)
    current_params["detector_threshold"] = best_thresh
    print(f"  => Selected detector_threshold: {best_thresh} (mAP: {best_map:.2f}%)\n")

    return current_params, history


def main():
    parser = argparse.ArgumentParser(description="Clean Validation Tuning for Vynix 1.0 and 2.0")
    parser.add_argument("--dataset-dir", default=r"E:\Dataset")
    parser.add_argument("--val-ids", default="review_response/val_10_ids.json")
    parser.add_argument("--model-v1-dir", default="saved_models_v1.0_train90")
    parser.add_argument("--model-v2-dir", default="saved_models_v2.0_train90")
    parser.add_argument("--val-cache-v1", default="review_response/val_caches/val_cache_v1.0.pt")
    parser.add_argument("--val-cache-v2", default="review_response/val_caches/val_cache_v2.0.pt")
    parser.add_argument("--results-json", default="review_response/val_tuning_results.json")
    parser.add_argument("--frozen-json", default="review_response/frozen_hyperparameters.json")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="all", choices=["all", "vynix_1.0", "vynix_2.0"])
    args = parser.parse_args()

    meta = HOIMeta(args.dataset_dir)
    all_history = []
    frozen_configs = {}

    if os.path.exists(args.results_json):
        try:
            with open(args.results_json, "r", encoding="utf-8") as f:
                all_history = json.load(f)
        except Exception:
            pass
    if os.path.exists(args.frozen_json):
        try:
            with open(args.frozen_json, "r", encoding="utf-8") as f:
                frozen_configs = json.load(f)
        except Exception:
            pass

    # 1. Process Vynix 1.0
    if args.model in ["all", "vynix_1.0"]:
        if not args.skip_extract or not os.path.exists(args.val_cache_v1):
            cached_v1 = extract_val_cache(
                dataset_dir=args.dataset_dir,
                model_dir=args.model_v1_dir,
                val_ids_path=args.val_ids,
                output_cache=args.val_cache_v1,
                detector_model="yolov8m.pt",
                detector_conf=0.05,
                limit=args.limit,
            )
        else:
            print(f"Loading existing Vynix 1.0 val cache from {args.val_cache_v1}...")
            cached_v1 = torch.load(args.val_cache_v1, map_location="cpu", weights_only=False)

        best_v1, hist_v1 = tune_vynix1(cached_v1, meta)
        # Load learned alpha/beta from train90 stats (do not tune)
        v1_stats_path = os.path.join(args.model_v1_dir, "training_stats.json")
        if os.path.exists(v1_stats_path):
            with open(v1_stats_path, "r", encoding="utf-8") as f:
                s = json.load(f)
                best_v1["alpha"] = round(float(s["learned_alpha"]), 4)
                best_v1["beta"] = round(float(s["learned_beta"]), 4)
        all_history = [h for h in all_history if h.get("model") != "vynix_1.0"] + hist_v1
        frozen_configs["vynix_1.0"] = best_v1

    # 2. Process Vynix 2.0
    if args.model in ["all", "vynix_2.0"]:
        if not args.skip_extract or not os.path.exists(args.val_cache_v2):
            cached_v2 = extract_val_cache(
                dataset_dir=args.dataset_dir,
                model_dir=args.model_v2_dir,
                val_ids_path=args.val_ids,
                output_cache=args.val_cache_v2,
                detector_model="yolov8x.pt",
                detector_conf=0.05,
                limit=args.limit,
            )
        else:
            print(f"Loading existing Vynix 2.0 val cache from {args.val_cache_v2}...")
            cached_v2 = torch.load(args.val_cache_v2, map_location="cpu", weights_only=False)

        best_v2, hist_v2 = tune_vynix2(cached_v2, meta)
        # Load learned alpha/beta from train90 stats (do not tune)
        v2_stats_path = os.path.join(args.model_v2_dir, "training_stats.json")
        if os.path.exists(v2_stats_path):
            with open(v2_stats_path, "r", encoding="utf-8") as f:
                s = json.load(f)
                best_v2["alpha"] = round(float(s["learned_alpha"]), 4)
                best_v2["beta"] = round(float(s["learned_beta"]), 4)
        all_history = [h for h in all_history if h.get("model") != "vynix_2.0"] + hist_v2
        frozen_configs["vynix_2.0"] = best_v2

    # Save metadata
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_hash = "unknown"

    import transformers
    import ultralytics

    metadata = {
        "random_seed": 42,
        "git_commit": git_hash,
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ultralytics": ultralytics.__version__,
            "numpy": np.__version__,
        },
        "selection_rule": "Highest val_10 Full mAP; tie-breaker: if within 0.1 mAP of peak, pick value closest to default. Alpha and beta kept as learned from train_90 without tuning.",
        "evaluation_protocol": "Averaged over classes with >=1 GT in val_10 (only_valid_gt=True).",
    }
    frozen_configs["metadata"] = metadata

    # Save all grid results
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n  ✓ Saved full grid tuning history ({len(all_history)} points) to {args.results_json}")

    # Save frozen hyperparameters
    with open(args.frozen_json, "w", encoding="utf-8") as f:
        json.dump(frozen_configs, f, indent=2)
    print(f"  ✓ Saved frozen hyperparameters to {args.frozen_json}")

    print("\n" + "=" * 80)
    print("  FINAL FROZEN HYPERPARAMETERS (Tuned on Validation Set Only):")
    print(json.dumps(frozen_configs, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
