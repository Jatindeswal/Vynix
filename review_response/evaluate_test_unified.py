#!/usr/bin/env python3
"""
Unified Single-Pass Test-Set Evaluation Runner for Project Vynix
===============================================================
Computes:
1. Box-level Default mAP (Full 600, Standard Rare 138, Standard Non-Rare 462, Legacy Pseudo-Rare 150)
2. Legacy Image-level AP (Full 600, Standard Rare 138, Standard Non-Rare 462, Legacy Pseudo-Rare 150)
3. Ablation variants in the SAME single pass:
   - Full pipeline (frozen hyperparameters)
   - No geometric gate (gamma_geo = 1.0)
   - Hard binary gate (IoU > 0 for contact verbs)
   - No object gate (gamma_obj = 0.0) [Vynix 2.0]
   - lambda = 1.0 [Vynix 2.0]
   - Raw baseline (no gates, lambda = 1.0)
   - YOLOv8x proposals + unaligned v1.0 cache [Vynix 2.0 pass]
4. Veto and rescue hypothesis audit:
   - Total hypotheses
   - Total soft vetoes (gate < 0.05 at frozen sigma and nominal sigma=0.08)
   - Total hard vetoes (IoU == 0 for contact verbs)
   - Split by GT-matched (box-level min(IoU_h, IoU_o)>=0.5 and image-level) vs unmatched FP
   - Rescued hypotheses (contact verb, IoU==0, d_box <= sigma, GT-matched)
5. Stage-wise wall-clock timing:
   - Detector (YOLO)
   - Visual Encoding (CLIP ViT-B/16)
   - Adapter & Scoring (MLP + Gates)
6. Per-class AP and per-class GT counts for the full test set.
7. Shard-level checkpointing and resumption.
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
from typing import Dict, List, Optional, Tuple, Set

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


def parse_raw_objects(raw_objs):
    if isinstance(raw_objs, str):
        return ast.literal_eval(raw_objs) if raw_objs.strip() else []
    return raw_objs if isinstance(raw_objs, list) else []


def evaluate_unified(
    model_name: str,  # "vynix_1.0" or "vynix_2.0"
    dataset_dir: str = r"E:\Dataset",
    output_dir: str = r"review_response",
    predictions_dir: str = r"E:\Do not open\Projects\Vynix\test_predictions",
    limit: Optional[int] = None,
    device: str = "cuda",
):
    print("=" * 95)
    print(f"  UNIFIED SINGLE-PASS TEST-SET EVALUATION: {model_name.upper()}")
    print("=" * 95)

    # 1. Load frozen hyperparameters
    frozen_json_path = os.path.join(output_dir, "frozen_hyperparameters.json")
    with open(frozen_json_path, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    hp = frozen_config[model_name]
    sigma = hp["sigma"]
    det_lambda = hp["lambda"]
    obj_gamma = hp["gamma_obj"]
    det_thresh = hp["detector_threshold"]

    if model_name == "vynix_1.0":
        detector_model = "yolov8m.pt"
        model_dir = "saved_models_v1.0"
        is_v2 = False
    else:
        detector_model = "yolov8x.pt"
        model_dir = "saved_models_v2.0"
        is_v2 = True

    print(f"  Detector         : {detector_model} (conf={det_thresh})")
    print(f"  Model Directory  : {model_dir}")
    print(f"  Frozen Params    : sigma={sigma}, lambda={det_lambda}, gamma_obj={obj_gamma}")
    print(f"  Device           : {device}")
    print("=" * 95)

    # 2. Metadata & Extractor
    meta = HOIMeta(dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=device,
        detector_conf=det_thresh,
        use_3stream=True,
        detector_model=detector_model,
        clip_model="openai/clip-vit-base-patch16",
    )
    extractor.yolo.to(device).eval()

    text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(device)
    obj_text_weights = extractor.build_object_text_weights(meta)

    # 3. Main Adapter
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
    print(f"  Main Adapter loaded: alpha={adapter.alpha.item():.4f}, beta={adapter.beta.item():.4f}")

    # For v2.0, also load v1.0 adapter for the intermediate "YOLOv8x + unaligned v1.0 cache" ablation
    adapter_v1 = None
    if is_v2:
        v1_cache_path = os.path.join("saved_models_v1.0", "vynix_full_cache.pt")
        v1_model_path = os.path.join("saved_models_v1.0", "vynix_full_model.pth")
        v1_cache_data = torch.load(v1_cache_path, map_location=device, weights_only=True)
        adapter_v1 = Vynix3StreamAdapter(
            cache_keys=v1_cache_data["cache_keys"],
            cache_values=v1_cache_data["cache_values"],
            use_spatial_mlp=True,
            num_classes=meta.num_classes,
        ).to(device).eval()
        if os.path.exists(v1_model_path):
            adapter_v1.load_state_dict(torch.load(v1_model_path, map_location=device, weights_only=True))
        print(f"  Secondary Adapter (v1.0) loaded for intermediate ablation.")

    # 4. Set up Evaluators for all variants
    variants = ["full", "no_geom_gate", "hard_binary_gate", "raw"]
    if is_v2:
        variants.extend(["no_obj_gate", "lambda_1.0", "yolov8x_v1_cache"])

    evaluators = {
        v: HICODetBoxEvaluator(
            num_classes=meta.num_classes,
            iou_thresh=0.5,
            split_json_path="review_response/standard_hico_splits.json",
        )
        for v in variants
    }

    # Predictions storage for disk persistence
    saved_ranked_predictions = {v: {} for v in variants}

    # 5. Veto & Rescue Counters
    veto_stats = {
        "total_hypotheses": 0,
        "contact_hypotheses": 0,
        "non_contact_hypotheses": 0,
        # Frozen sigma vetoes
        "soft_vetoes_frozen": 0,
        "soft_vetoes_frozen_box_matched": 0,
        "soft_vetoes_frozen_img_matched": 0,
        "soft_vetoes_frozen_unmatched": 0,
        # Nominal sigma=0.08 vetoes (for comparison with paper)
        "soft_vetoes_sigma08": 0,
        "soft_vetoes_sigma08_box_matched": 0,
        "soft_vetoes_sigma08_img_matched": 0,
        "soft_vetoes_sigma08_unmatched": 0,
        # Hard binary vetoes
        "hard_vetoes": 0,
        "hard_vetoes_box_matched": 0,
        "hard_vetoes_img_matched": 0,
        "hard_vetoes_unmatched": 0,
        # Rescued hypotheses (IoU==0, d_box <= sigma, and GT-matched)
        "rescued_frozen_box_matched": 0,
        "rescued_frozen_img_matched": 0,
        "rescued_sigma08_box_matched": 0,
        "rescued_sigma08_img_matched": 0,
    }

    # Stage timing accumulators
    stage_times = {
        "detector_sec": 0.0,
        "clip_sec": 0.0,
        "adapter_scoring_sec": 0.0,
        "eval_overhead_sec": 0.0,
    }

    # Discover test files
    test_files = sorted(glob.glob(os.path.join(dataset_dir, "data", "test-*.parquet")))
    total_test_rows = sum(pq.read_metadata(f).num_rows for f in test_files)
    n_target = min(total_test_rows, limit) if limit else total_test_rows
    print(f"  Target Test Images: {n_target} across {len(test_files)} parquet files.\n")

    os.makedirs(predictions_dir, exist_ok=True)
    t0 = time.time()
    processed_count = 0

    with tqdm(total=n_target, desc=f"  Test Eval ({model_name})", unit="img") as pbar:
        for shard_idx, f in enumerate(test_files):
            shard_pred_path = os.path.join(predictions_dir, f"preds_{model_name}_shard_{shard_idx:02d}.pt")
            
            # Check for existing checkpoint resumption (only if full shard was saved and not doing a limit run)
            if not limit and os.path.exists(shard_pred_path):
                print(f"\n  [Resume] Loading pre-existing Shard {shard_idx:02d} from {shard_pred_path}...")
                shard_data = torch.load(shard_pred_path, map_location="cpu", weights_only=False)
                shard_preds = shard_data["predictions"]
                shard_gts = shard_data["ground_truths"]
                shard_vetoes = shard_data.get("veto_stats", {})
                shard_st = shard_data.get("stage_times", {})

                for k, v in shard_vetoes.items():
                    veto_stats[k] += v
                for k, v in shard_st.items():
                    stage_times[k] += v

                # Replay into evaluators
                pf = pq.ParquetFile(f)
                num_in_shard = pf.metadata.num_rows
                for rel_idx in range(num_in_shard):
                    eval_idx = processed_count
                    processed_count += 1
                    raw_objs = shard_gts.get(eval_idx, [])
                    for ev in evaluators.values():
                        ev.add_ground_truth(eval_idx, raw_objs)
                    for v_name in variants:
                        triplets = shard_preds[v_name].get(eval_idx, [])
                        evaluators[v_name].add_predictions(eval_idx, triplets)
                        saved_ranked_predictions[v_name][eval_idx] = triplets
                    pbar.update(1)
                print(f"  ✓ Replayed {num_in_shard} images from Shard {shard_idx:02d}. Total processed: {processed_count}")
                continue

            # Fresh extraction for this shard
            pf = pq.ParquetFile(f)
            reached_limit = False
            shard_raw_gts = {}
            shard_preds_dict = {v: {} for v in variants}
            shard_veto_delta = {k: 0 for k in veto_stats}
            shard_stage_delta = {k: 0.0 for k in stage_times}

            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if limit and processed_count >= limit:
                        reached_limit = True
                        break

                    eval_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    raw_objs = parse_raw_objects(row.get("objects", []))
                    shard_raw_gts[eval_idx] = raw_objs
                    t_ev0 = time.time()
                    for ev in evaluators.values():
                        ev.add_ground_truth(eval_idx, raw_objs)
                    stage_times["eval_overhead_sec"] += (time.time() - t_ev0)

                    # Build quick GT lookup for veto audit
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

                    # Parse Image
                    img_data = row["image"]
                    if isinstance(img_data, dict) and img_data.get("bytes") is not None:
                        pil_img = Image.open(io.BytesIO(img_data["bytes"]))
                    elif isinstance(img_data, Image.Image):
                        pil_img = img_data
                    else:
                        continue
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    img_w, img_h = pil_img.size

                    # Stage 1: Detector
                    t_det0 = time.time()
                    persons, objects = extractor.detect(pil_img)
                    dt_det = time.time() - t_det0
                    stage_times["detector_sec"] += dt_det
                    shard_stage_delta["detector_sec"] += dt_det

                    if not persons or not objects:
                        del pil_img
                        continue

                    # Filter pairs by det_thresh and top-50
                    pairs = [(p, o) for p in persons for o in objects if p[1] >= det_thresh and o[1] >= det_thresh]
                    if len(pairs) > 50:
                        pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
                        pairs = pairs[:50]

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
                        del pil_img
                        continue

                    # Stage 2: CLIP Feature Extraction
                    t_clip0 = time.time()
                    boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)
                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch
                    dt_clip = time.time() - t_clip0
                    stage_times["clip_sec"] += dt_clip
                    shard_stage_delta["clip_sec"] += dt_clip

                    # Stage 3: Adapter Scoring & Gate Rescoring
                    t_adapt0 = time.time()
                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights.T
                        final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
                        probs_batch = torch.softmax(final_logits, dim=1).cpu()

                        probs_v1_batch = None
                        if is_v2 and adapter_v1 is not None:
                            final_logits_v1 = adapter_v1(clip_logits, f_fused_batch, s_vecs_batch)
                            probs_v1_batch = torch.softmax(final_logits_v1, dim=1).cpu()

                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

                    del pil_img

                    # Build candidate predictions per variant
                    variant_triplets = {v: [] for v in variants}

                    for k_idx, v in enumerate(valid_pairs):
                        p_conf, o_conf = v["p_conf"], v["o_conf"]
                        iou_val, b_dist = v["iou_val"], v["b_dist"]
                        p_box, o_box = v["p_box"], v["o_box"]

                        # Object semantic gate
                        cos_sim = 1.0
                        if v["entries"]:
                            hico_name = meta.hoi_to_obj[v["entries"][0][2]]
                            if hico_name in obj_text_weights:
                                t_obj = obj_text_weights[hico_name].cpu()
                                cos_sim = max(0.0, (f_objs_norm[k_idx] @ t_obj).item())

                        det_score_tuned = (p_conf * o_conf) ** det_lambda
                        det_score_1 = (p_conf * o_conf) ** 1.00
                        obj_gate_tuned = (cos_sim ** obj_gamma) if obj_gamma > 0 else 1.0

                        for verb, _, hoi_id in v["entries"]:
                            raw_prob = probs_batch[k_idx, hoi_id].item()

                            # Compute gates
                            gate_frozen = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma)
                            gate_sigma08 = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=0.08)
                            gate_hard = 0.0 if (iou_val == 0.0 and gate_frozen < 1.0) else 1.0

                            is_contact_verb = (compute_soft_geometric_gate(verb, 0.0, 0.1, sigma_contact=0.08) < 1.0)

                            # --- Veto and Rescue Tracking ---
                            veto_stats["total_hypotheses"] += 1
                            shard_veto_delta["total_hypotheses"] += 1
                            if is_contact_verb:
                                veto_stats["contact_hypotheses"] += 1
                                shard_veto_delta["contact_hypotheses"] += 1
                            else:
                                veto_stats["non_contact_hypotheses"] += 1
                                shard_veto_delta["non_contact_hypotheses"] += 1

                            # Check GT matches for this hypothesis
                            box_matched = False
                            for (gt_h, gt_o, gt_hid) in gt_box_list:
                                if gt_hid == hoi_id and compute_iou(p_box, gt_h) >= 0.5 and compute_iou(o_box, gt_o) >= 0.5:
                                    box_matched = True
                                    break
                            img_matched = (hoi_id in gt_img_hois)

                            # Frozen soft veto (< 0.05)
                            if gate_frozen < 0.05:
                                veto_stats["soft_vetoes_frozen"] += 1
                                shard_veto_delta["soft_vetoes_frozen"] += 1
                                if box_matched:
                                    veto_stats["soft_vetoes_frozen_box_matched"] += 1
                                    shard_veto_delta["soft_vetoes_frozen_box_matched"] += 1
                                if img_matched:
                                    veto_stats["soft_vetoes_frozen_img_matched"] += 1
                                    shard_veto_delta["soft_vetoes_frozen_img_matched"] += 1
                                if not img_matched:
                                    veto_stats["soft_vetoes_frozen_unmatched"] += 1
                                    shard_veto_delta["soft_vetoes_frozen_unmatched"] += 1

                            # Nominal sigma=0.08 soft veto (< 0.05)
                            if gate_sigma08 < 0.05:
                                veto_stats["soft_vetoes_sigma08"] += 1
                                shard_veto_delta["soft_vetoes_sigma08"] += 1
                                if box_matched:
                                    veto_stats["soft_vetoes_sigma08_box_matched"] += 1
                                    shard_veto_delta["soft_vetoes_sigma08_box_matched"] += 1
                                if img_matched:
                                    veto_stats["soft_vetoes_sigma08_img_matched"] += 1
                                    shard_veto_delta["soft_vetoes_sigma08_img_matched"] += 1
                                if not img_matched:
                                    veto_stats["soft_vetoes_sigma08_unmatched"] += 1
                                    shard_veto_delta["soft_vetoes_sigma08_unmatched"] += 1

                            # Hard veto (iou == 0 for contact verb)
                            if is_contact_verb and iou_val == 0.0:
                                veto_stats["hard_vetoes"] += 1
                                shard_veto_delta["hard_vetoes"] += 1
                                if box_matched:
                                    veto_stats["hard_vetoes_box_matched"] += 1
                                    shard_veto_delta["hard_vetoes_box_matched"] += 1
                                if img_matched:
                                    veto_stats["hard_vetoes_img_matched"] += 1
                                    shard_veto_delta["hard_vetoes_img_matched"] += 1
                                if not img_matched:
                                    veto_stats["hard_vetoes_unmatched"] += 1
                                    shard_veto_delta["hard_vetoes_unmatched"] += 1

                            # Rescued hypotheses (contact verb, IoU==0, d_box <= sigma, and GT-matched)
                            if is_contact_verb and iou_val == 0.0:
                                if b_dist <= sigma:
                                    if box_matched:
                                        veto_stats["rescued_frozen_box_matched"] += 1
                                        shard_veto_delta["rescued_frozen_box_matched"] += 1
                                    if img_matched:
                                        veto_stats["rescued_frozen_img_matched"] += 1
                                        shard_veto_delta["rescued_frozen_img_matched"] += 1
                                if b_dist <= 0.08:
                                    if box_matched:
                                        veto_stats["rescued_sigma08_box_matched"] += 1
                                        shard_veto_delta["rescued_sigma08_box_matched"] += 1
                                    if img_matched:
                                        veto_stats["rescued_sigma08_img_matched"] += 1
                                        shard_veto_delta["rescued_sigma08_img_matched"] += 1

                            # --- Candidate Scoring for Variants ---
                            # 1. Full
                            conf_full = det_score_tuned * raw_prob * gate_frozen * obj_gate_tuned
                            variant_triplets["full"].append((p_box, o_box, hoi_id, conf_full))

                            # 2. No Geometric Gate
                            conf_no_geo = det_score_tuned * raw_prob * 1.0 * obj_gate_tuned
                            variant_triplets["no_geom_gate"].append((p_box, o_box, hoi_id, conf_no_geo))

                            # 3. Hard Binary Gate
                            conf_hard = det_score_tuned * raw_prob * gate_hard * obj_gate_tuned
                            variant_triplets["hard_binary_gate"].append((p_box, o_box, hoi_id, conf_hard))

                            # 4. Raw Baseline
                            conf_raw = det_score_1 * raw_prob * 1.0 * 1.0
                            variant_triplets["raw"].append((p_box, o_box, hoi_id, conf_raw))

                            # V2.0 specific variants
                            if is_v2:
                                # 5. No Object Gate
                                conf_no_obj = det_score_tuned * raw_prob * gate_frozen * 1.0
                                variant_triplets["no_obj_gate"].append((p_box, o_box, hoi_id, conf_no_obj))

                                # 6. Lambda = 1.0
                                conf_lam1 = det_score_1 * raw_prob * gate_frozen * obj_gate_tuned
                                variant_triplets["lambda_1.0"].append((p_box, o_box, hoi_id, conf_lam1))

                                # 7. YOLOv8x proposals + unaligned v1.0 cache
                                if probs_v1_batch is not None:
                                    raw_prob_v1 = probs_v1_batch[k_idx, hoi_id].item()
                                    conf_v1_cache = det_score_1 * raw_prob_v1 * gate_frozen * 1.0
                                    variant_triplets["yolov8x_v1_cache"].append((p_box, o_box, hoi_id, conf_v1_cache))

                    dt_adapt = time.time() - t_adapt0
                    stage_times["adapter_scoring_sec"] += dt_adapt
                    shard_stage_delta["adapter_scoring_sec"] += dt_adapt

                    # Feed predictions to all evaluators
                    t_ev1 = time.time()
                    for v_name in variants:
                        evaluators[v_name].add_predictions(eval_idx, variant_triplets[v_name])
                        saved_ranked_predictions[v_name][eval_idx] = variant_triplets[v_name]
                        shard_preds_dict[v_name][eval_idx] = variant_triplets[v_name]
                    stage_times["eval_overhead_sec"] += (time.time() - t_ev1)

                if reached_limit:
                    break

            # Save intermediate checkpoint for this shard
            shard_payload = {
                "shard_idx": shard_idx,
                "predictions": shard_preds_dict,
                "ground_truths": shard_raw_gts,
                "veto_stats": shard_veto_delta,
                "stage_times": shard_stage_delta,
            }
            torch.save(shard_payload, shard_pred_path)
            print(f"\n  [Checkpoint] Saved Shard {shard_idx:02d} ({len(shard_raw_gts)} imgs) -> {shard_pred_path}")

            if reached_limit:
                break

    dt = time.time() - t0
    fps = processed_count / max(dt, 1)
    print(f"\n" + "=" * 95)
    print(f"  INFERENCE FINISHED: {processed_count} images in {dt/60:.2f} min ({fps:.2f} img/s).")
    print("=" * 95)

    # Save full predictions to disk
    full_pred_path = os.path.join(predictions_dir, f"test_predictions_{model_name}_full.pt")
    torch.save(saved_ranked_predictions, full_pred_path)
    print(f"  ✓ Saved all ranked predictions to {full_pred_path}")

    # Compute metrics for all variants
    all_metrics = {}
    print("\nComputing Box-Level and Image-Level metrics for all variants...")

    pseudo_rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    pseudo_non_rare_ids = set(range(meta.num_classes)) - pseudo_rare_ids

    # Ground truth coverage
    main_evaluator = evaluators["full"]
    gt_counts = {c: int(main_evaluator.num_gt_per_class[c]) for c in range(meta.num_classes)}
    classes_with_gt = sum(1 for c, cnt in gt_counts.items() if cnt > 0)
    rare_with_gt = sum(1 for c in main_evaluator.rare_ids if gt_counts.get(c, 0) > 0)
    non_rare_with_gt = sum(1 for c in main_evaluator.non_rare_ids if gt_counts.get(c, 0) > 0)

    print(f"\n  [Test Ground-Truth Coverage]")
    print(f"    Full: {classes_with_gt}/600 classes have >=1 GT instance")
    print(f"    Rare 138: {rare_with_gt}/138 classes have >=1 GT instance")
    print(f"    Non-Rare 462: {non_rare_with_gt}/462 classes have >=1 GT instance")

    for v_name in variants:
        ev = evaluators[v_name]
        b_res = ev.evaluate_box_level()
        i_res = ev.evaluate_image_level(processed_count)

        box_p_rare = float(np.mean([b_res["per_class_ap"][h] for h in pseudo_rare_ids])) * 100.0
        box_p_nonrare = float(np.mean([b_res["per_class_ap"][h] for h in pseudo_non_rare_ids])) * 100.0
        img_p_rare = float(np.mean([i_res["per_class_ap"][h] for h in pseudo_rare_ids])) * 100.0
        img_p_nonrare = float(np.mean([i_res["per_class_ap"][h] for h in pseudo_non_rare_ids])) * 100.0

        all_metrics[v_name] = {
            "box_level_default": {
                "mAP_full": b_res["mAP_full"],
                "mAP_rare_138": b_res["mAP_rare"],
                "mAP_non_rare_462": b_res["mAP_non_rare"],
                "mAP_pseudo_rare_150": box_p_rare,
                "mAP_pseudo_non_rare_450": box_p_nonrare,
            },
            "image_level_legacy": {
                "mAP_full": i_res["mAP_full"],
                "mAP_rare_138": i_res["mAP_rare"],
                "mAP_non_rare_462": i_res["mAP_non_rare"],
                "mAP_pseudo_rare_150": img_p_rare,
                "mAP_pseudo_non_rare_450": img_p_nonrare,
            },
            "per_class_box_ap": {int(c): float(b_res["per_class_ap"][c]) for c in range(meta.num_classes)},
            "per_class_img_ap": {int(c): float(i_res["per_class_ap"][c]) for c in range(meta.num_classes)},
        }

    # Summary Output Table
    print("\n" + "=" * 115)
    print(f"  FINAL TEST-SET EVALUATION RESULTS — {model_name.upper()} ({processed_count} images)")
    print("=" * 115)
    print(f"  {'Ablation Variant':<30} | {'Box Full':<10} | {'Box Rare138':<12} | {'Box NonRare':<12} | {'Img Full':<10} | {'Img Rare138':<12} | {'Img NonRare':<12}")
    print("-" * 115)
    for v_name in variants:
        bm = all_metrics[v_name]["box_level_default"]
        im = all_metrics[v_name]["image_level_legacy"]
        print(f"  {v_name:<30} | {bm['mAP_full']:<10.2f} | {bm['mAP_rare_138']:<12.2f} | {bm['mAP_non_rare_462']:<12.2f} | {im['mAP_full']:<10.2f} | {im['mAP_rare_138']:<12.2f} | {im['mAP_non_rare_462']:<12.2f}")
    print("=" * 115)

    print("\n--- Veto and Rescue Audit ---")
    for k, v in veto_stats.items():
        print(f"  {k:<35}: {v:,}")

    # Stage timing breakdown
    n_img = max(processed_count, 1)
    stage_breakdown = {
        "detector": {
            "total_min": stage_times["detector_sec"] / 60.0,
            "ms_per_img": (stage_times["detector_sec"] / n_img) * 1000.0,
            "pct": (stage_times["detector_sec"] / max(dt, 1e-5)) * 100.0,
        },
        "clip_encode": {
            "total_min": stage_times["clip_sec"] / 60.0,
            "ms_per_img": (stage_times["clip_sec"] / n_img) * 1000.0,
            "pct": (stage_times["clip_sec"] / max(dt, 1e-5)) * 100.0,
        },
        "adapter_scoring": {
            "total_min": stage_times["adapter_scoring_sec"] / 60.0,
            "ms_per_img": (stage_times["adapter_scoring_sec"] / n_img) * 1000.0,
            "pct": (stage_times["adapter_scoring_sec"] / max(dt, 1e-5)) * 100.0,
        },
        "eval_overhead": {
            "total_min": stage_times["eval_overhead_sec"] / 60.0,
            "ms_per_img": (stage_times["eval_overhead_sec"] / n_img) * 1000.0,
            "pct": (stage_times["eval_overhead_sec"] / max(dt, 1e-5)) * 100.0,
        },
        "total_inference": {
            "total_min": dt / 60.0,
            "ms_per_img": (dt / n_img) * 1000.0,
            "fps": fps,
        },
    }

    print("\n--- Stage-Wise Wall-Clock Timing ---")
    print(f"  {'Stage':<20} | {'Total Time':<12} | {'Per-Image':<12} | {'Share (%)':<10}")
    print("-" * 60)
    for stg in ["detector", "clip_encode", "adapter_scoring", "eval_overhead"]:
        info = stage_breakdown[stg]
        print(f"  {stg:<20} | {info['total_min']:>8.2f} min | {info['ms_per_img']:>8.1f} ms | {info['pct']:>8.1f}%")
    print("-" * 60)
    tot = stage_breakdown["total_inference"]
    print(f"  {'Total Wall-Clock':<20} | {tot['total_min']:>8.2f} min | {tot['ms_per_img']:>8.1f} ms | (100.0%)")

    # Build final report dictionary
    final_output = {
        "model": model_name,
        "detector": detector_model,
        "images_evaluated": processed_count,
        "wall_clock_minutes": dt / 60.0,
        "fps": fps,
        "frozen_hyperparameters": hp,
        "gt_coverage": {
            "total_classes": meta.num_classes,
            "classes_with_gt": classes_with_gt,
            "rare_138_with_gt": rare_with_gt,
            "non_rare_462_with_gt": non_rare_with_gt,
            "per_class_gt_counts": gt_counts,
        },
        "stage_times": stage_breakdown,
        "metrics_by_variant": all_metrics,
        "veto_and_rescue_audit": veto_stats,
    }

    final_json_path = os.path.join(output_dir, f"final_test_results_{model_name}.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved final test results to {final_json_path}")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Unified Single-Pass Test Evaluator")
    parser.add_argument("--model", choices=["vynix_1.0", "vynix_2.0"], required=True)
    parser.add_argument("--dataset-dir", default=r"E:\Dataset")
    parser.add_argument("--output-dir", default=r"review_response")
    parser.add_argument("--predictions-dir", default=r"E:\Do not open\Projects\Vynix\test_predictions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_unified(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        predictions_dir=args.predictions_dir,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
