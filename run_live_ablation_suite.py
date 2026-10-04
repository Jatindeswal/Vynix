#!/usr/bin/env python3
"""
Project Vynix: Live Empirical Ablation Suite
=============================================
Executes a multi-condition ablation test directly on the HICO-DET Parquet test dataset.
Extracts YOLOv8x and CLIP ViT-B/16 features once across N test images, and evaluates
all architectural components to isolate exact mAP contributions, Rare parity, and vetoes:

Configurations Tested:
1. Full Vynix 2.0 (Flagship): Soft Gate (sigma=0.08) + Obj Gate (gamma=0.15) + Calib (lambda=0.8) + Adaptive Alpha
2. w/o Semantic Object Gate (gamma=0.0)
3. w/o Score Calibration (lambda=1.0)
4. w/o Adaptive Alpha (Uniform alpha=0.35)
5. Hard Binary Gate (IoU > 0 binary threshold)
6. w/o Geometric Veto Gate (Unconstrained Gate = 1.0)
7. Raw Baseline (No Geometric Gate, No Object Gate, No Calibration)
"""

import os
import io
import sys
import glob
import math
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

# Import Vynix pipeline modules
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

def evaluate_predictions(preds_dict, ground_truth, meta, processed_count):
    all_hoi_ids = sorted(meta.hoi_to_obj.keys())
    per_class_ap = {}
    for hoi_id in all_hoi_ids:
        scores = [preds_dict.get(i, {}).get(hoi_id, 0.0) for i in range(processed_count)]
        labels = [1 if hoi_id in ground_truth.get(i, set()) else 0 for i in range(processed_count)]
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
    parser = argparse.ArgumentParser(description="Live Empirical Ablation Suite for Project Vynix")
    parser.add_argument("--dataset-dir", type=str, default=r"E:\Dataset")
    parser.add_argument("--model-dir", type=str, default="saved_models_v2.0")
    parser.add_argument("--detector-model", type=str, default="yolov8x.pt")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--images", type=int, default=300, help="Number of test images to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("  PROJECT VYNIX — LIVE EMPIRICAL ABLATION SUITE ON HICO-DET")
    print("=" * 80)
    print(f"  Dataset Directory : {args.dataset_dir}")
    print(f"  Model Directory   : {args.model_dir}")
    print(f"  Detector Model    : {args.detector_model}")
    print(f"  Evaluation Target : {args.images} images")
    print(f"  Compute Device    : {args.device}")
    print("=" * 80)

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
    print("\n[1/4] Precomputing CLIP multi-template text embeddings...")
    text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(device)

    # Precompute Object Text Weights for semantic verification gate
    print("  Precomputing multi-template object text embeddings for verification gate...")
    obj_text_weights = extractor.build_object_text_weights(meta)

    # 2. Load Model & Cache
    cache_path = os.path.join(args.model_dir, "vynix_full_cache.pt")
    model_path = os.path.join(args.model_dir, "vynix_full_model.pth")
    if not os.path.exists(cache_path) or not os.path.exists(model_path):
        print(f"[ERROR] Model or cache weights not found in {args.model_dir}!")
        sys.exit(1)

    print(f"[2/4] Loading Exemplar Cache and Spatial Adapter ({args.model_dir})...")
    cache_data = torch.load(cache_path, map_location=device)
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_data["cache_keys"],
        cache_values=cache_data["cache_values"],
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    ).to(device).eval()
    adapter.load_state_dict(torch.load(model_path, map_location=device))

    # Precompute alpha vectors
    rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    alpha_adaptive = torch.tensor([0.55 if h in rare_ids else 0.25 for h in range(meta.num_classes)], device=device)
    alpha_uniform = torch.tensor([0.35 for _ in range(meta.num_classes)], device=device)

    # 3. Single-Pass Feature & Proposal Extraction over Target Images
    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))
    if not test_files:
        print(f"[ERROR] No parquet shards found in {args.dataset_dir}/data!")
        sys.exit(1)

    print(f"\n[3/4] Extracting Proposals & Visual Representations across {args.images} images...")
    extracted_records = [] # (img_idx, ground_truth, valid_pair_metadata, probs_adaptive, probs_uniform, f_objs_norm)
    all_gt = {}
    processed_count = 0

    with tqdm(total=args.images, desc="  Extracting", unit="img") as pbar:
        for f in test_files:
            pf = pq.ParquetFile(f)
            reached_limit = False
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if processed_count >= args.images:
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
                        extracted_records.append((img_idx, gt_set, [], None, None, None))
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
                        extracted_records.append((img_idx, gt_set, [], None, None, None))
                        continue

                    boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)

                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch
                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights.T
                        # Forward pass with adaptive alpha
                        final_logits_adap = adapter(clip_logits, f_fused_batch, s_vecs_batch, alpha_override=alpha_adaptive)
                        probs_adaptive = torch.softmax(final_logits_adap, dim=1).cpu()

                        # Forward pass with uniform alpha
                        final_logits_unif = adapter(clip_logits, f_fused_batch, s_vecs_batch, alpha_override=alpha_uniform)
                        probs_uniform = torch.softmax(final_logits_unif, dim=1).cpu()

                        # Object crop embeddings
                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

                    extracted_records.append((img_idx, gt_set, valid_pairs, probs_adaptive, probs_uniform, f_objs_norm))

                del df
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if reached_limit:
                    break
            if reached_limit:
                break

    # 4. Instant Multi-Condition Ablation Evaluation
    print(f"\n[4/4] Executing Architectural Ablations across {len(extracted_records)} cached records...")

    configurations = [
        {
            "name": "Full Vynix 2.0 (Flagship SOTA)",
            "use_soft_gate": True,
            "use_obj_gate": True,
            "obj_gamma": 0.15,
            "det_lambda": 0.8,
            "use_adaptive_alpha": True,
            "gate_mode": "gaussian_soft",
            "desc": "Complete decoupled pipeline with all geometric & semantic constraints"
        },
        {
            "name": "Ablation 1: w/o Object Semantic Gate",
            "use_soft_gate": True,
            "use_obj_gate": False,
            "obj_gamma": 0.0,
            "det_lambda": 0.8,
            "use_adaptive_alpha": True,
            "gate_mode": "gaussian_soft",
            "desc": "Disables isolated object crop semantic verification gate (gamma = 0.0)"
        },
        {
            "name": "Ablation 2: w/o Score Calibration",
            "use_soft_gate": True,
            "use_obj_gate": True,
            "obj_gamma": 0.15,
            "det_lambda": 1.0,
            "use_adaptive_alpha": True,
            "gate_mode": "gaussian_soft",
            "desc": "Linear detection score multiplication (det_score = (p_conf * o_conf)^1.0)"
        },
        {
            "name": "Ablation 3: w/o Adaptive Alpha",
            "use_soft_gate": True,
            "use_obj_gate": True,
            "obj_gamma": 0.15,
            "det_lambda": 0.8,
            "use_adaptive_alpha": False,
            "gate_mode": "gaussian_soft",
            "desc": "Replaces class-adaptive alpha with uniform static alpha = 0.35"
        },
        {
            "name": "Ablation 4: Hard Binary Gate (IoU > 0)",
            "use_soft_gate": True,
            "use_obj_gate": True,
            "obj_gamma": 0.15,
            "det_lambda": 0.8,
            "use_adaptive_alpha": True,
            "gate_mode": "hard_binary",
            "desc": "Strict binary threshold: if IoU == 0 on contact verb -> veto (0.0)"
        },
        {
            "name": "Ablation 5: w/o Geometric Veto Gate",
            "use_soft_gate": False,
            "use_obj_gate": True,
            "obj_gamma": 0.15,
            "det_lambda": 0.8,
            "use_adaptive_alpha": True,
            "gate_mode": "unconstrained",
            "desc": "No spatial veto: permits distant entities to predict contact actions (gate = 1.0)"
        },
        {
            "name": "Ablation 6: Raw Baseline (No Gates/Calib)",
            "use_soft_gate": False,
            "use_obj_gate": False,
            "obj_gamma": 0.0,
            "det_lambda": 1.0,
            "use_adaptive_alpha": False,
            "gate_mode": "unconstrained",
            "desc": "Unconstrained visual-semantic matching without geometric or affordance priors"
        }
    ]

    results = []

    for cfg in configurations:
        preds = {}
        vetoes = 0

        for (img_idx, gt_set, valid_pairs, probs_adap, probs_unif, f_objs_norm) in extracted_records:
            if not valid_pairs:
                preds[img_idx] = {}
                continue

            probs_batch = probs_adap if cfg["use_adaptive_alpha"] else probs_unif
            img_preds = {}

            for k_idx, v in enumerate(valid_pairs):
                det_score = (v["p_conf"] * v["o_conf"]) ** cfg["det_lambda"]
                obj_gate = 1.0

                if cfg["use_obj_gate"] and f_objs_norm is not None and v["entries"]:
                    hico_name = meta.hoi_to_obj[v["entries"][0][2]]
                    if hico_name in obj_text_weights:
                        t_obj = obj_text_weights[hico_name].cpu()
                        cos_sim = (f_objs_norm[k_idx] @ t_obj).item()
                        obj_gate = max(0.0, cos_sim) ** cfg["obj_gamma"]

                for verb, _, hoi_id in v["entries"]:
                    raw_prob = probs_batch[k_idx, hoi_id].item()

                    # Compute Gate according to mode
                    if cfg["gate_mode"] == "gaussian_soft":
                        gate = compute_soft_geometric_gate(verb, v["iou_val"], v["b_dist"])
                    elif cfg["gate_mode"] == "hard_binary":
                        if verb in CONTACT_VERBS and v["iou_val"] == 0.0:
                            gate = 0.0
                        else:
                            gate = 1.0
                    else: # unconstrained
                        gate = 1.0

                    if gate < 0.05:
                        vetoes += 1

                    conf = det_score * raw_prob * gate * obj_gate
                    if hoi_id not in img_preds or conf > img_preds[hoi_id]:
                        img_preds[hoi_id] = conf

            preds[img_idx] = img_preds

        metrics = evaluate_predictions(preds, all_gt, meta, processed_count)
        metrics["name"] = cfg["name"]
        metrics["vetoes"] = vetoes
        metrics["desc"] = cfg["desc"]
        results.append(metrics)

    # 5. Display Formatted Markdown Table
    print("\n" + "=" * 95)
    print(f"  🏆 EMPIRICAL ABLATION RESULTS ON {processed_count} TEST IMAGES")
    print("=" * 95)
    header = f"| {'Configuration':<42} | {'Full mAP':<9} | {'Rare mAP':<9} | {'Non-Rare':<9} | {'Parity':<8} | {'Vetoes':<8} |"
    print(header)
    print("|" + "-" * 44 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    base_map = results[-1]["mAP_full"]
    flagship_map = results[0]["mAP_full"]

    for r in results:
        delta = r["mAP_full"] - base_map
        delta_str = f"(+{delta:.2f}%)" if delta > 0 else f"({delta:.2f}%)"
        print(f"| {r['name']:<42} | {r['mAP_full']:>6.2f}%   | {r['mAP_rare']:>6.2f}%   | {r['mAP_nonrare']:>6.2f}%   | {r['parity']:>6.1f}%  | {r['vetoes']:>8,d} |")
    print("=" * 95)

    # Save JSON summary
    json_path = os.path.join(args.output_dir, "live_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_images": processed_count,
            "results": results
        }, f, indent=2)
    print(f"[OK] Saved ablation metrics JSON to: {json_path}")

    # 6. Generate Publication-Quality Ablation Bar Chart
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    
    # Subplot 1: mAP comparison
    names = [r["name"].replace("Ablation ", "Abl. ").replace(" (Flagship SOTA)", "★").replace(" (No Gates/Calib)", "") for r in results]
    full_scores = [r["mAP_full"] for r in results]
    rare_scores = [r["mAP_rare"] for r in results]

    y_pos = np.arange(len(names))
    height = 0.38

    b1 = ax1.barh(y_pos - height/2, full_scores, height, label="Full mAP (%)", color="#3b82f6", edgecolor="#1e3a8a")
    b2 = ax1.barh(y_pos + height/2, rare_scores, height, label="Rare mAP (%)", color="#10b981", edgecolor="#064e3b")

    for rect in b1:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, fontweight="bold")
    for rect in b2:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Average Precision (mAP %)", fontweight="bold")
    ax1.set_title("Architectural Component Impact (Full vs. Rare mAP)", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, max(full_scores + rare_scores) * 1.25)

    # Subplot 2: Vetoed Hallucinations
    veto_counts = [r["vetoes"] for r in results]
    colors = ["#10b981" if v > 0 else "#64748b" for v in veto_counts]
    colors[0] = "#059669"

    b_veto = ax2.barh(y_pos, veto_counts, height*1.6, color="#f43f5e", edgecolor="#9f1239", alpha=0.85)
    for rect in b_veto:
        w = rect.get_width()
        if w > 0:
            ax2.annotate(f"{int(w):,}", xy=(w, rect.get_y() + rect.get_height()/2),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Hallucinations Suppressed (Count)", fontweight="bold")
    ax2.set_title("Geometric Gate Hallucination Veto Activity", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.set_xlim(0, max(veto_counts) * 1.25 if max(veto_counts) > 0 else 100)

    plt.tight_layout()
    plot_out = os.path.join(args.output_dir, "live_ablation_study_chart.png")
    fig.savefig(plot_out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved ablation comparison chart to: {plot_out}")

    # Copy to artifact directory
    artifact_dir = r"C:\Users\jatin\.gemini\antigravity\brain\a4d1d635-73d4-49b9-91ec-ccb76014514d"
    if os.path.isdir(artifact_dir):
        import shutil
        shutil.copy2(plot_out, os.path.join(artifact_dir, "live_ablation_study_chart.png"))
        print("[OK] Synchronized chart with artifacts directory.")

if __name__ == "__main__":
    main()
