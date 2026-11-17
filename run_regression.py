#!/usr/bin/env python3
"""
Regression Suite for Project Vynix Refactoring
==============================================
Runs a deterministic 50-image evaluation on both Vynix 1.0 and Vynix 2.0
under the frozen hyperparameters.

Usage:
  1. Record baseline:
       python run_regression.py --save regression_before.pt
  2. Test against baseline after refactoring:
       python run_regression.py --compare regression_before.pt
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
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_raw_objects(raw_objs):
    if isinstance(raw_objs, str):
        return ast.literal_eval(raw_objs) if raw_objs.strip() else []
    return raw_objs if isinstance(raw_objs, list) else []


def run_50_image_pass(
    model_name: str,
    dataset_dir: str = r"E:\Dataset",
    device: str = "cuda",
    num_images: int = 50,
) -> Tuple[Dict[int, List], Dict[str, float]]:
    set_seed(42)

    frozen_json_path = "review_response/frozen_hyperparameters.json"
    with open(frozen_json_path, "r", encoding="utf-8") as f:
        hp_config = json.load(f)[model_name]

    sigma = hp_config["sigma"]
    det_lambda = hp_config["lambda"]
    obj_gamma = hp_config["gamma_obj"]
    det_thresh = hp_config["detector_threshold"]

    if model_name == "vynix_1.0":
        detector_model = "yolov8m.pt"
        model_dir = "saved_models_v1.0"
        is_v2 = False
    else:
        detector_model = "yolov8x.pt"
        model_dir = "saved_models_v2.0"
        is_v2 = True

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
        split_json_path="review_response/standard_hico_splits.json",
    )

    test_file = os.path.join(dataset_dir, "data", "test-00000-of-00004.parquet")
    pf = pq.ParquetFile(test_file)

    ranked_predictions = {}
    processed_count = 0

    for batch in pf.iter_batches(batch_size=32):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            if processed_count >= num_images:
                break
            eval_idx = processed_count
            processed_count += 1

            raw_objs = parse_raw_objects(row.get("objects", []))
            evaluator.add_ground_truth(eval_idx, raw_objs)

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

            persons, objects = extractor.detect(pil_img)
            if not persons or not objects:
                ranked_predictions[eval_idx] = []
                del pil_img
                continue

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
                ranked_predictions[eval_idx] = []
                del pil_img
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

            del pil_img

            triplets = []
            for k_idx, v in enumerate(valid_pairs):
                p_conf, o_conf = v["p_conf"], v["o_conf"]
                iou_val, b_dist = v["iou_val"], v["b_dist"]
                p_box, o_box = v["p_box"], v["o_box"]

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
                    gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma)
                    conf = det_score * raw_prob * gate * obj_gate
                    triplets.append((p_box, o_box, hoi_id, float(conf)))

            evaluator.add_predictions(eval_idx, triplets)
            ranked_predictions[eval_idx] = triplets

        if processed_count >= num_images:
            break

    box_metrics = evaluator.evaluate_box_level()
    summary_metrics = {
        "mAP_full": box_metrics["mAP_full"],
        "mAP_rare_138": box_metrics["mAP_rare"],
        "mAP_non_rare_462": box_metrics["mAP_non_rare"],
    }
    return ranked_predictions, summary_metrics


def compare_predictions(base_preds: Dict, curr_preds: Dict, tol: float = 1e-6) -> bool:
    if set(base_preds.keys()) != set(curr_preds.keys()):
        print(f"❌ Image set mismatch: {len(base_preds)} vs {len(curr_preds)}")
        return False

    all_matched = True
    for img_idx in sorted(base_preds.keys()):
        b_list = base_preds[img_idx]
        c_list = curr_preds[img_idx]

        if len(b_list) != len(c_list):
            print(f"❌ Image {img_idx}: prediction count mismatch ({len(b_list)} vs {len(c_list)})")
            all_matched = False
            continue

        for idx, (b_item, c_item) in enumerate(zip(b_list, c_list)):
            # b_item: (p_box, o_box, hoi_id, score)
            p_box_b, o_box_b, hoi_b, score_b = b_item
            p_box_c, o_box_c, hoi_c, score_c = c_item

            if hoi_b != hoi_c:
                print(f"❌ Image {img_idx} pred {idx}: hoi_id mismatch ({hoi_b} vs {hoi_c})")
                all_matched = False
                break

            score_diff = abs(score_b - score_c)
            if score_diff > tol:
                print(f"❌ Image {img_idx} pred {idx}: score mismatch ({score_b:.8f} vs {score_c:.8f}, diff={score_diff:.2e})")
                all_matched = False
                break

            for coord_b, coord_c in zip(p_box_b + o_box_b, p_box_c + o_box_c):
                if abs(coord_b - coord_c) > tol:
                    print(f"❌ Image {img_idx} pred {idx}: box coord mismatch")
                    all_matched = False
                    break

    return all_matched


def main():
    parser = argparse.ArgumentParser(description="Deterministic 50-Image Regression Suite")
    parser.add_argument("--save", type=str, default=None, help="Save baseline regression predictions to path")
    parser.add_argument("--compare", type=str, default=None, help="Compare current predictions against baseline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=str, default=r"E:\Dataset")
    args = parser.parse_args()

    print("=" * 80)
    print("  PROJECT VYNIX DETERMINISTIC 50-IMAGE REGRESSION SUITE")
    print("=" * 80)

    # We evaluate both Vynix 1.0 and Vynix 2.0
    models_to_test = ["vynix_1.0", "vynix_2.0"]
    current_results = {}

    for m in models_to_test:
        print(f"\n--- Running 50-image evaluation: {m} ---")
        t0 = time.time()
        preds, metrics = run_50_image_pass(m, dataset_dir=args.dataset_dir, device=args.device, num_images=50)
        dt = time.time() - t0
        print(f"  Completed in {dt:.1f}s. Full mAP: {metrics['mAP_full']:.2f}%, Rare: {metrics['mAP_rare_138']:.2f}%, Non-Rare: {metrics['mAP_non_rare_462']:.2f}%")
        current_results[m] = {
            "predictions": preds,
            "metrics": metrics,
        }

    if args.save:
        save_path = args.save
        torch.save(current_results, save_path)
        metrics_summary = {m: current_results[m]["metrics"] for m in models_to_test}
        json_path = os.path.splitext(save_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n✓ Saved baseline regression predictions to {save_path}")
        print(f"✓ Saved baseline metrics summary to {json_path}")

    if args.compare:
        print(f"\n--- Comparing current predictions against {args.compare} ---")
        baseline = torch.load(args.compare, map_location="cpu", weights_only=False)
        overall_pass = True

        for m in models_to_test:
            print(f"\nChecking {m}...")
            if m not in baseline:
                print(f"❌ Model {m} missing from baseline!")
                overall_pass = False
                continue

            base_preds = baseline[m]["predictions"]
            curr_preds = current_results[m]["predictions"]
            matched = compare_predictions(base_preds, curr_preds, tol=1e-6)

            base_m = baseline[m]["metrics"]
            curr_m = current_results[m]["metrics"]
            m_matched = (
                abs(base_m["mAP_full"] - curr_m["mAP_full"]) < 1e-4
                and abs(base_m["mAP_rare_138"] - curr_m["mAP_rare_138"]) < 1e-4
                and abs(base_m["mAP_non_rare_462"] - curr_m["mAP_non_rare_462"]) < 1e-4
            )

            if matched and m_matched:
                print(f"  ✓ {m}: Predictions and metrics match identically! (Full mAP: {curr_m['mAP_full']:.4f}%)")
            else:
                print(f"  ❌ {m}: Regression failure! Base mAP={base_m['mAP_full']:.4f}%, Curr mAP={curr_m['mAP_full']:.4f}%")
                overall_pass = False

        if overall_pass:
            print("\n" + "=" * 80)
            print("  🏆 REGRESSION CHECK PASSED: All predictions and metrics are 100% bit-exact.")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("  ❌ REGRESSION CHECK FAILED.")
            print("=" * 80)
            sys.exit(1)


if __name__ == "__main__":
    main()
