#!/usr/bin/env python3
"""
Clean 300-Image Stage Timing Benchmark for Project Vynix
========================================================
Measures accurate per-stage inference times on 300 fixed test images (seed 42)
with GPU synchronization and no concurrent GPU processes:
- Object Detection (YOLO)
- Visual Feature Encoding (CLIP ViT-B/16)
- Adapter & Gate Rescoring (Spatial MLP, Text projection, Soft gate)
Repeated twice for both Vynix 1.0 and Vynix 2.0.
"""

import ast
import io
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_benchmark_pass(
    model_name: str,
    dataset_dir: str = r"E:\Dataset",
    num_images: int = 300,
    device: str = "cuda",
) -> Dict:
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
    else:
        detector_model = "yolov8x.pt"
        model_dir = "saved_models_v2.0"

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

    test_file = os.path.join(dataset_dir, "data", "test-00000-of-00004.parquet")
    pf = pq.ParquetFile(test_file)

    # Warmup on 5 dummy images
    dummy_img = Image.new("RGB", (640, 640), (128, 128, 128))
    for _ in range(5):
        p, o = extractor.detect(dummy_img)
    torch.cuda.synchronize()

    t_det_total = 0.0
    t_clip_total = 0.0
    t_adapt_total = 0.0

    processed_count = 0
    t0_wall = time.time()

    for batch in pf.iter_batches(batch_size=32):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            if processed_count >= num_images:
                break
            processed_count += 1

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
            torch.cuda.synchronize()
            t_s = time.time()
            persons, objects = extractor.detect(pil_img)
            torch.cuda.synchronize()
            t_det_total += (time.time() - t_s)

            if not persons or not objects:
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
                del pil_img
                continue

            # Stage 2: CLIP Feature Extraction
            torch.cuda.synchronize()
            t_s = time.time()
            boxes_to_extract = [(v["p_box"], v["o_box"], v["u_box"]) for v in valid_pairs]
            f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
            s_vecs_batch = torch.stack([v["s_vec"] for v in valid_pairs]).to(device)
            f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch
            torch.cuda.synchronize()
            t_clip_total += (time.time() - t_s)

            # Stage 3: Adapter Scoring & Gate Rescoring
            torch.cuda.synchronize()
            t_s = time.time()
            with torch.no_grad():
                clip_logits = f_unions @ text_weights.T
                final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
                probs_batch = torch.softmax(final_logits, dim=1).cpu()

                f_objs = f_fused_batch[:, 512:1024]
                f_objs_norm = (f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)).cpu()

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

            torch.cuda.synchronize()
            t_adapt_total += (time.time() - t_s)
            del pil_img

        if processed_count >= num_images:
            break

    total_wall_sec = time.time() - t0_wall
    n = max(processed_count, 1)

    result = {
        "model": model_name,
        "detector": detector_model,
        "num_images": processed_count,
        "wall_clock_sec": total_wall_sec,
        "fps": n / total_wall_sec,
        "ms_per_img": {
            "detector": (t_det_total / n) * 1000.0,
            "clip_encode": (t_clip_total / n) * 1000.0,
            "adapter_scoring": (t_adapt_total / n) * 1000.0,
            "total_measured": (total_wall_sec / n) * 1000.0,
        },
        "stage_totals_sec": {
            "detector": t_det_total,
            "clip_encode": t_clip_total,
            "adapter_scoring": t_adapt_total,
        },
        "shares_pct": {
            "detector": (t_det_total / total_wall_sec) * 100.0,
            "clip_encode": (t_clip_total / total_wall_sec) * 100.0,
            "adapter_scoring": (t_adapt_total / total_wall_sec) * 100.0,
        },
        "extrapolation_9658_min": {
            "detector": (t_det_total / n) * 9658.0 / 60.0,
            "clip_encode": (t_clip_total / n) * 9658.0 / 60.0,
            "adapter_scoring": (t_adapt_total / n) * 9658.0 / 60.0,
            "total": (total_wall_sec / n) * 9658.0 / 60.0,
        },
    }
    return result


def main():
    print("=" * 80)
    print("  PROJECT VYNIX CLEAN 300-IMAGE TIMING BENCHMARK (Repeated Twice)")
    print("=" * 80)

    benchmark_records = {}

    for model in ["vynix_1.0", "vynix_2.0"]:
        benchmark_records[model] = {}
        for run_id in ["run_1", "run_2"]:
            print(f"\n>>> Running {model} [{run_id}] on 300 test images (seed 42)...")
            res = run_benchmark_pass(model, num_images=300)
            benchmark_records[model][run_id] = res
            ms = res["ms_per_img"]
            sh = res["shares_pct"]
            ext = res["extrapolation_9658_min"]
            print(f"    Total Wall-Clock: {res['wall_clock_sec']:.1f}s ({res['fps']:.2f} img/s, {ms['total_measured']:.1f} ms/img)")
            print(f"      - Detector    : {ms['detector']:>6.1f} ms/img ({sh['detector']:>4.1f}%) | Extrapolated 9,658: {ext['detector']:>5.2f} min")
            print(f"      - CLIP Encode : {ms['clip_encode']:>6.1f} ms/img ({sh['clip_encode']:>4.1f}%) | Extrapolated 9,658: {ext['clip_encode']:>5.2f} min")
            print(f"      - Adapter     : {ms['adapter_scoring']:>6.1f} ms/img ({sh['adapter_scoring']:>4.1f}%) | Extrapolated 9,658: {ext['adapter_scoring']:>5.2f} min")
            print(f"      - Total Extrapolated (9,658 imgs): {ext['total']:>5.2f} min (~{ext['total']/60:.2f} GPU-h)")

    out_file = "review_response/clean_timing_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f"\n✓ Saved clean timing benchmark records to {out_file}")


if __name__ == "__main__":
    main()
