#!/usr/bin/env python3
"""
Project Vynix: Full Test Set Benchmark & Multi-Stream Empirical Ablation Suite
=============================================================================
Evaluates all 9,658 official HICO-DET test images across 4 Parquet shards.
Extracts YOLOv8x and CLIP ViT-B/16 visual-spatial representations ONCE in a single pass,
and simultaneously maintains 6 parallel evaluation streams to isolate the exact empirical
impact of every architectural component across the ENTIRE test set.

Streams Evaluated:
1. [Flagship SOTA Baseline]: Soft Gate (sigma=0.08) + Obj Gate (gamma=0.10) + Calib (lambda=0.90)
2. [Ablation 1: w/o Geometric Gate]: Spatial Gate = 1.0 (unconstrained, measuring false-positive penalty)
3. [Ablation 2: Hard Binary Gate]: IoU > 0 strict binary threshold (measuring over-veto penalty)
4. [Ablation 3: w/o Object Gate]: Obj Gate gamma = 0.0 (measuring object semantic verification)
5. [Ablation 4: w/o Score Calib]: lambda = 1.00 (measuring linear vs sublinear calibration)
6. [Ablation 5: Raw Baseline]: Gate = 1.0, Obj Gate = 1.0, lambda = 1.00 (completely unconstrained)
"""

import os
import io
import sys
import glob
import math
import time
import json
import argparse
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from vynix_fewshot_adapter import (
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    compute_iou,
    compute_box_distance,
    compute_union_box,
    compute_spatial_vector,
    compute_soft_geometric_gate,
    compute_ap,
    normalize_name,
    parse_int_list,
    COCO_CLASSES,
    CONTACT_VERBS
)

def evaluate_stream(all_preds, all_gt, meta, total_images):
    all_hoi_ids = sorted(meta.hoi_to_obj.keys())
    per_class_ap = {}
    for hoi_id in all_hoi_ids:
        scores = [all_preds.get(i, {}).get(hoi_id, 0.0) for i in range(total_images)]
        labels = [1 if hoi_id in all_gt.get(i, set()) else 0 for i in range(total_images)]
        per_class_ap[hoi_id] = compute_ap(scores, labels)

    full_aps = [per_class_ap[h] for h in all_hoi_ids]
    rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    nonrare_ids = set(range(meta.num_classes)) - rare_ids
    rare_aps = [per_class_ap[h] for h in rare_ids if h in per_class_ap]
    nonrare_aps = [per_class_ap[h] for h in nonrare_ids if h in per_class_ap]

    map_full = float(np.mean(full_aps)) * 100
    map_rare = float(np.mean(rare_aps)) * 100 if rare_aps else 0.0
    map_nonrare = float(np.mean(nonrare_aps)) * 100 if nonrare_aps else 0.0
    parity = (map_rare / map_nonrare * 100) if map_nonrare > 0 else 0.0

    return {
        "mAP_full": map_full,
        "mAP_rare": map_rare,
        "mAP_nonrare": map_nonrare,
        "parity": parity
    }

def main():
    parser = argparse.ArgumentParser(description="Full Test Set Benchmark & Multi-Stream Ablation Suite")
    parser.add_argument("--dataset-dir", type=str, default=r"E:\Dataset")
    parser.add_argument("--model-dir", type=str, default="saved_models_yolov8x_vitb16")
    parser.add_argument("--detector-model", type=str, default="yolov8x.pt")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit (default: None for all 9658 images)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 85)
    print("  PROJECT VYNIX — FULL TEST SET BENCHMARK & MULTI-STREAM ABLATION SUITE")
    print("=" * 85)
    print(f"  Dataset Directory : {args.dataset_dir}")
    print(f"  Model Directory   : {args.model_dir}")
    print(f"  Detector Model    : {args.detector_model}")
    print(f"  CLIP Backbone     : {args.clip_model}")
    print(f"  Device            : {args.device}")
    print("=" * 85)

    # 1. Initialize Metadata and Extractors
    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=args.device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector_model,
        clip_model=args.clip_model
    )
    extractor.yolo.to(device).eval()

    # Precompute CLIP text embeddings
    print("\n[1/5] Precomputing CLIP multi-template text embeddings...")
    text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(device)

    # Precompute Object Text Weights
    print("  Precomputing multi-template object text embeddings for verification gate...")
    obj_text_weights = extractor.build_object_text_weights(meta)

    # 2. Load Model & Cache
    cache_path = os.path.join(args.model_dir, "vynix_full_cache.pt")
    model_path = os.path.join(args.model_dir, "vynix_full_model.pth")
    if not os.path.exists(cache_path) or not os.path.exists(model_path):
        print(f"[ERROR] Model or cache weights not found in {args.model_dir}!")
        sys.exit(1)

    print(f"\n[2/5] Loading Exemplar Cache and Spatial Adapter ({args.model_dir})...")
    cache_data = torch.load(cache_path, map_location=device)
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_data["cache_keys"],
        cache_values=cache_data["cache_values"],
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    ).to(device).eval()
    adapter.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Discover Test Shards
    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))
    if not test_files:
        print(f"[ERROR] No parquet shards found in {args.dataset_dir}/data!")
        sys.exit(1)

    total_images_available = sum(pq.read_metadata(f).num_rows for f in test_files)
    n_target = min(total_images_available, args.limit) if args.limit else total_images_available
    print(f"\n[3/5] Discovered {len(test_files)} test shards ({total_images_available} total images). Target: {n_target} images.")

    # 4. Stream Definitions
    streams = [
        {"id": "flagship_baseline", "name": "Flagship Vynix 2.0 (SOTA)", "gate": "soft", "gamma": 0.10, "lam": 0.90, "desc": "Full flagship model (Soft Gate, Obj Gate, lambda=0.9)"},
        {"id": "no_geom_gate",     "name": "w/o Geometric Veto Gate",  "gate": "none", "gamma": 0.10, "lam": 0.90, "desc": "Spatial gate disabled (gate=1.0 for all pairs)"},
        {"id": "hard_binary_gate", "name": "Hard Binary Gate (IoU>0)",  "gate": "hard", "gamma": 0.10, "lam": 0.90, "desc": "Strict IoU>0 binary threshold on contact verbs"},
        {"id": "no_obj_gate",      "name": "w/o Object Semantic Gate",  "gate": "soft", "gamma": 0.00, "lam": 0.90, "desc": "Object verification disabled (gamma=0.0)"},
        {"id": "no_score_calib",   "name": "w/o Score Calibration",     "gate": "soft", "gamma": 0.10, "lam": 1.00, "desc": "Linear detection score (lambda=1.0)"},
        {"id": "raw_baseline",     "name": "Raw Baseline (No Gates)",   "gate": "none", "gamma": 0.00, "lam": 1.00, "desc": "No geometric gate, no obj gate, lambda=1.0"}
    ]

    all_preds_streams = {s["id"]: {} for s in streams}
    veto_counts = {s["id"]: 0 for s in streams}
    all_gt = {}
    processed_count = 0

    print("\n[4/5] Launching Single-Pass Inference across all 6 streams...")
    t_start = time.time()

    with tqdm(total=n_target, desc="  Evaluating", unit="img") as pbar:
        for f in test_files:
            pf = pq.ParquetFile(f)
            reached_limit = False
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if n_target and processed_count >= n_target:
                        reached_limit = True
                        break
                    img_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    gt_set = set(parse_int_list(str(row.get("positive_objects", ""))))
                    all_gt[img_idx] = gt_set

                    img_data = row["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
                        pil_img = Image.open(io.BytesIO(img_data["bytes"]))
                    elif isinstance(img_data, Image.Image):
                        pil_img = img_data
                    else:
                        continue
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    persons, objects = extractor.detect(pil_img)
                    if not persons or not objects:
                        for s in streams:
                            all_preds_streams[s["id"]][img_idx] = {}
                        continue

                    img_w, img_h = pil_img.size
                    pairs = [(p, o) for p in persons for o in objects]
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
                            "entries": entries
                        })

                    if not valid_pairs:
                        for s in streams:
                            all_preds_streams[s["id"]][img_idx] = {}
                        continue

                    boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)

                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch
                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights.T
                        # Model evaluation with trained adapter
                        final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch, alpha_override=None)
                        probs_batch = torch.softmax(final_logits, dim=1).cpu()

                        # Object crop embeddings
                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

                    # Compute predictions for all 6 streams simultaneously
                    stream_img_preds = {s["id"]: {} for s in streams}

                    for k_idx, v in enumerate(valid_pairs):
                        p_conf, o_conf = v["p_conf"], v["o_conf"]
                        iou_val, b_dist = v["iou_val"], v["b_dist"]

                        # Precompute object similarity
                        cos_sim = 1.0
                        if f_objs_norm is not None and v["entries"]:
                            hico_name = meta.hoi_to_obj[v["entries"][0][2]]
                            if hico_name in obj_text_weights:
                                t_obj = obj_text_weights[hico_name].cpu()
                                cos_sim = max(0.0, (f_objs_norm[k_idx] @ t_obj).item())

                        for verb, _, hoi_id in v["entries"]:
                            raw_prob = probs_batch[k_idx, hoi_id].item()

                            # Compute gate factors
                            soft_gate = compute_soft_geometric_gate(verb, iou_val, b_dist)
                            hard_gate = 0.0 if (verb in CONTACT_VERBS and iou_val == 0.0) else 1.0

                            # Populate each stream
                            for s in streams:
                                s_id = s["id"]
                                det_score = (p_conf * o_conf) ** s["lam"]
                                obj_mult = (cos_sim ** s["gamma"]) if s["gamma"] > 0 else 1.0

                                if s["gate"] == "soft":
                                    gate_val = soft_gate
                                elif s["gate"] == "hard":
                                    gate_val = hard_gate
                                else:
                                    gate_val = 1.0

                                if gate_val < 0.05:
                                    veto_counts[s_id] += 1

                                conf = det_score * raw_prob * gate_val * obj_mult
                                if hoi_id not in stream_img_preds[s_id] or conf > stream_img_preds[s_id][hoi_id]:
                                    stream_img_preds[s_id][hoi_id] = conf

                    for s in streams:
                        all_preds_streams[s["id"]][img_idx] = stream_img_preds[s["id"]]

                    try:
                        del pil_img
                    except NameError:
                        pass

                del df
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if reached_limit:
                    break
            if reached_limit:
                break

    dt_total = time.time() - t_start
    print(f"\n[INFO] Inference completed across {processed_count} images in {dt_total/60:.2f} minutes ({processed_count/max(dt_total, 1):.2f} img/s).")

    # 5. Compute Final Benchmarks & Metrics for All Streams
    print("\n[5/5] Calculating Average Precision & Parity across all 600 classes...")
    final_results = []
    for s in streams:
        metrics = evaluate_stream(all_preds_streams[s["id"]], all_gt, meta, processed_count)
        metrics["id"] = s["id"]
        metrics["name"] = s["name"]
        metrics["desc"] = s["desc"]
        metrics["vetoes"] = veto_counts[s["id"]]
        final_results.append(metrics)

    # 6. Print Official Results Table
    print("\n" + "=" * 98)
    print(f"  🏆 OFFICIAL BENCHMARK & ABLATION RESULTS (FULL TEST SET: {processed_count} IMAGES)")
    print("=" * 98)
    header = f"| {'Configuration':<38} | {'Full mAP':<9} | {'Rare mAP':<9} | {'Non-Rare':<9} | {'Parity':<8} | {'Vetoes':<9} |"
    print(header)
    print("|" + "-" * 40 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 11 + "|")

    for r in final_results:
        is_flagship = (r["id"] == "flagship_baseline")
        prefix = "★ " if is_flagship else "  "
        print(f"| {prefix + r['name']:<38} | {r['mAP_full']:>6.2f}%   | {r['mAP_rare']:>6.2f}%   | {r['mAP_nonrare']:>6.2f}%   | {r['parity']:>6.1f}%  | {r['vetoes']:>9,d} |")
    print("=" * 98)

    # Save to JSON
    json_path = os.path.join(args.output_dir, "full_test_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_test_images": processed_count,
            "elapsed_minutes": dt_total / 60.0,
            "results": final_results
        }, f, indent=2)
    print(f"\n[OK] Saved final results JSON to: {json_path}")

    # Generate Publication Chart
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": "#1e293b",
        "axes.linewidth": 1.1,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    names = [r["name"] for r in final_results]
    full_scores = [r["mAP_full"] for r in final_results]
    rare_scores = [r["mAP_rare"] for r in final_results]
    y_pos = np.arange(len(names))
    height = 0.36

    b1 = ax1.barh(y_pos - height/2, full_scores, height, label="Full mAP (%)", color="#3b82f6", edgecolor="#1e3a8a")
    b2 = ax1.barh(y_pos + height/2, rare_scores, height, label="Rare mAP (%)", color="#10b981", edgecolor="#064e3b")

    for rect in b1:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")
    for rect in b2:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Average Precision (mAP %)", fontweight="bold")
    ax1.set_title(f"HICO-DET Full Test Set Ablation Benchmark ({processed_count} Images)", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, max(full_scores + rare_scores) * 1.25)

    # Vetoes chart
    veto_counts_arr = [r["vetoes"] for r in final_results]
    b_veto = ax2.barh(y_pos, veto_counts_arr, height*1.6, color="#f43f5e", edgecolor="#9f1239", alpha=0.85)
    for rect in b_veto:
        w = rect.get_width()
        if w > 0:
            ax2.annotate(f"{int(w):,}", xy=(w, rect.get_y() + rect.get_height()/2),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Hallucinations Suppressed (Count)", fontweight="bold")
    ax2.set_title("Geometric Hallucination Veto Activity", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.set_xlim(0, max(veto_counts_arr) * 1.25 if max(veto_counts_arr) > 0 else 100)

    plt.tight_layout()
    chart_out = os.path.join(args.output_dir, "full_test_ablation_benchmark.png")
    fig.savefig(chart_out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved full test ablation chart to: {chart_out}")

    # Copy to artifacts
    artifact_dir = r"C:\Users\jatin\.gemini\antigravity\brain\a4d1d635-73d4-49b9-91ec-ccb76014514d"
    if os.path.isdir(artifact_dir):
        import shutil
        shutil.copy2(chart_out, os.path.join(artifact_dir, "full_test_ablation_benchmark.png"))
        shutil.copy2(json_path, os.path.join(artifact_dir, "full_test_ablation_results.json"))
        print("[OK] Synchronized deliverables with artifact directory.")

if __name__ == "__main__":
    main()
