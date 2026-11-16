#!/usr/bin/env python3
"""
Project Vynix: Standard Box-Level HICO-DET Evaluation Runner
============================================================
Evaluates Vynix under the official HICO-DET Default setting:
- Prediction triplet: (human_box, object_box, hoi_id, score)
- True positive iff: min(IoU_h, IoU_o) >= 0.5 with unmatched GT
- Reports both new Box-Level mAP and old Image-Level AP side-by-side:
    * Full (600 classes)
    * Standard Rare (138 classes, <10 training instances)
    * Standard Non-Rare (462 classes, >=10 training instances)
    * Legacy Pseudo-Rare (150 classes, h % 4 == 0) for backward comparison
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


def run_evaluation(
    dataset_dir: str,
    model_dir: str,
    detector_model: str = "yolov8x.pt",
    clip_model: str = "openai/clip-vit-base-patch16",
    device: str = "cuda",
    split: str = "test",
    split_indices_path: Optional[str] = None,
    limit: Optional[int] = None,
    det_lambda: float = 0.90,
    obj_gamma: float = 0.10,
    sigma_contact: float = 0.08,
    use_hard_gate: bool = False,
    disable_gate: bool = False,
    split_json_path: str = "review_response/standard_hico_splits.json",
    output_json: Optional[str] = None,
    detector_conf: float = 0.08,
    save_predictions: Optional[str] = None,
):
    print("=" * 80)
    print("  PROJECT VYNIX — BOX-LEVEL & IMAGE-LEVEL HICO-DET EVALUATION")
    print("=" * 80)
    print(f"  Split            : {split}")
    print(f"  Detector         : {detector_model}")
    print(f"  CLIP Backbone    : {clip_model}")
    print(f"  det_lambda       : {det_lambda}")
    print(f"  obj_gamma        : {obj_gamma}")
    print(f"  sigma_contact    : {sigma_contact}")
    print(f"  Device           : {device}")
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

    # Load Adapter
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

    evaluator = HICODetBoxEvaluator(
        num_classes=meta.num_classes,
        iou_thresh=0.5,
        split_json_path=split_json_path,
    )

    # Discover parquet files
    if split == "test":
        files = sorted(glob.glob(os.path.join(dataset_dir, "data", "test-*.parquet")))
    else:
        files = sorted(glob.glob(os.path.join(dataset_dir, "data", "train-*.parquet")))

    target_indices = None
    if split_indices_path and os.path.exists(split_indices_path):
        with open(split_indices_path, "r", encoding="utf-8") as f:
            target_indices = set(json.load(f))
        print(f"  Loaded {len(target_indices)} target image indices from {split_indices_path}")

    total_available = sum(pq.read_metadata(f).num_rows for f in files)
    n_target = len(target_indices) if target_indices else total_available
    if limit:
        n_target = min(n_target, limit)
    print(f"  Evaluating on {n_target} images...")

    processed_count = 0
    global_img_idx = 0
    all_predictions = {}
    t0 = time.time()

    with tqdm(total=n_target, desc="  Evaluating", unit="img") as pbar:
        for f in files:
            pf = pq.ParquetFile(f)
            reached_limit = False
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    curr_idx = global_img_idx
                    global_img_idx += 1

                    if target_indices is not None and curr_idx not in target_indices:
                        continue
                    if limit and processed_count >= limit:
                        reached_limit = True
                        break

                    eval_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    # Parse GT objects
                    raw_objs = row.get("objects", [])
                    if isinstance(raw_objs, str):
                        raw_objs = ast.literal_eval(raw_objs) if raw_objs.strip() else []
                    evaluator.add_ground_truth(eval_idx, raw_objs)

                    # Image
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
                            "entries": entries,
                        })

                    if not valid_pairs:
                        continue

                    boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)
                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch

                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights.T
                        final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
                        probs_batch = torch.softmax(final_logits, dim=1).cpu()

                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

                    predicted_triplets = []
                    for k_idx, v in enumerate(valid_pairs):
                        p_conf, o_conf = v["p_conf"], v["o_conf"]
                        iou_val, b_dist = v["iou_val"], v["b_dist"]
                        p_box, o_box = v["p_box"], v["o_box"]

                        # Object semantic identity verification
                        cos_sim = 1.0
                        if v["entries"]:
                            hico_name = meta.hoi_to_obj[v["entries"][0][2]]
                            if hico_name in obj_text_weights:
                                t_obj = obj_text_weights[hico_name].cpu()
                                cos_sim = max(0.0, (f_objs_norm[k_idx] @ t_obj).item())

                        det_score = (p_conf * o_conf) ** det_lambda
                        obj_gate = (cos_sim ** obj_gamma) if obj_gamma > 0 else 1.0

                        for verb, _, hoi_id in v["entries"]:
                            raw_prob = probs_batch[k_idx, hoi_id].item()

                            if disable_gate:
                                gate = 1.0
                            elif use_hard_gate:
                                gate = 0.0 if (verb in meta.obj_to_entries and iou_val == 0.0) else 1.0
                            else:
                                gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma_contact)

                            conf = det_score * raw_prob * gate * obj_gate
                            predicted_triplets.append((p_box, o_box, hoi_id, conf))

                    evaluator.add_predictions(eval_idx, predicted_triplets)
                    all_predictions[eval_idx] = predicted_triplets

                if reached_limit:
                    break
            if reached_limit:
                break

    dt = time.time() - t0
    print(f"\nInference completed in {dt/60:.2f} minutes ({processed_count/max(dt,1):.2f} img/s).")

    if save_predictions:
        os.makedirs(os.path.dirname(save_predictions), exist_ok=True)
        torch.save(all_predictions, save_predictions)
        print(f"  ✓ Saved raw ranked predictions to {save_predictions}")

    # Compute both metrics
    print("\nComputing Box-Level Default mAP...")
    box_res = evaluator.evaluate_box_level()

    print("Computing Legacy Image-Level AP...")
    img_res = evaluator.evaluate_image_level(processed_count)

    # Legacy pseudo-rare (h % 4 == 0) for backward comparison
    pseudo_rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    pseudo_non_rare_ids = set(range(meta.num_classes)) - pseudo_rare_ids
    box_p_rare = float(np.mean([box_res["per_class_ap"][h] for h in pseudo_rare_ids])) * 100.0
    box_p_nonrare = float(np.mean([box_res["per_class_ap"][h] for h in pseudo_non_rare_ids])) * 100.0
    img_p_rare = float(np.mean([img_res["per_class_ap"][h] for h in pseudo_rare_ids])) * 100.0
    img_p_nonrare = float(np.mean([img_res["per_class_ap"][h] for h in pseudo_non_rare_ids])) * 100.0

    print("\n" + "=" * 95)
    print("  🏆 SIDE-BY-SIDE HICO-DET EVALUATION RESULTS")
    print("=" * 95)
    print(f"  {'Metric Category':<35} | {'Box-Level Default mAP (%)':<26} | {'Legacy Image-Level AP (%)':<26}")
    print("-" * 95)
    print(f"  {'Full (600 classes)':<35} | {box_res['mAP_full']:<26.2f} | {img_res['mAP_full']:<26.2f}")
    print(f"  {'Standard Rare (138 classes)':<35} | {box_res['mAP_rare']:<26.2f} | {img_res['mAP_rare']:<26.2f}")
    print(f"  {'Standard Non-Rare (462 classes)':<35} | {box_res['mAP_non_rare']:<26.2f} | {img_res['mAP_non_rare']:<26.2f}")
    print("-" * 95)
    print(f"  {'[Legacy] Pseudo-Rare (150 classes)':<35} | {box_p_rare:<26.2f} | {img_p_rare:<26.2f}")
    print(f"  {'[Legacy] Pseudo-Non-Rare (450)':<35} | {box_p_nonrare:<26.2f} | {img_p_nonrare:<26.2f}")
    print("=" * 95)

    out_data = {
        "split": split,
        "detector": detector_model,
        "clip_model": clip_model,
        "images_evaluated": processed_count,
        "elapsed_minutes": dt / 60.0,
        "parameters": {
            "det_lambda": det_lambda,
            "obj_gamma": obj_gamma,
            "sigma_contact": sigma_contact,
            "disable_gate": disable_gate,
            "use_hard_gate": use_hard_gate,
        },
        "box_level_default": {
            "mAP_full": box_res["mAP_full"],
            "mAP_rare_138": box_res["mAP_rare"],
            "mAP_non_rare_462": box_res["mAP_non_rare"],
            "mAP_pseudo_rare_150": box_p_rare,
            "mAP_pseudo_non_rare_450": box_p_nonrare,
        },
        "image_level_legacy": {
            "mAP_full": img_res["mAP_full"],
            "mAP_rare_138": img_res["mAP_rare"],
            "mAP_non_rare_462": img_res["mAP_non_rare"],
            "mAP_pseudo_rare_150": img_p_rare,
            "mAP_pseudo_non_rare_450": img_p_nonrare,
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nSaved evaluation results to {output_json}")

    return out_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vynix on Box-Level Default HICO-DET")
    parser.add_argument("--dataset-dir", default=r"E:\Dataset")
    parser.add_argument("--model-dir", default="saved_models_v2.0")
    parser.add_argument("--detector-model", default="yolov8x.pt")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--split-indices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--det-lambda", type=float, default=0.90)
    parser.add_argument("--obj-gamma", type=float, default=0.10)
    parser.add_argument("--sigma-contact", type=float, default=0.08)
    parser.add_argument("--detector-conf", type=float, default=0.08)
    parser.add_argument("--use-hard-gate", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--output-json", default="review_response/eval_box_results.json")
    parser.add_argument("--save-predictions", default=None, help="Path to save raw ranked predictions dict to disk (.pt)")
    args = parser.parse_args()

    run_evaluation(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        detector_model=args.detector_model,
        clip_model=args.clip_model,
        device=args.device,
        split=args.split,
        split_indices_path=args.split_indices,
        limit=args.limit,
        det_lambda=args.det_lambda,
        obj_gamma=args.obj_gamma,
        sigma_contact=args.sigma_contact,
        use_hard_gate=args.use_hard_gate,
        disable_gate=args.disable_gate,
        output_json=args.output_json,
        detector_conf=args.detector_conf,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()
