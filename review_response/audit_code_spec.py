#!/usr/bin/env python3
"""
Audit script for CODE_SPEC verification:
1. Inspects saved checkpoints for v1.0 and v2.0 (all parameters, buffers, shapes).
2. Runs standalone step-by-step scoring vs actual pipeline on 20 test pairs for v1.0 and v2.0.
3. Computes max absolute difference.
"""

import io
import json
import os
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import (
    COCO_CLASSES,
    HOIMeta,
    MultiStreamFeatureExtractor,
    SpatialMLP,
    Vynix3StreamAdapter,
    compute_box_distance,
    compute_iou,
    compute_soft_geometric_gate,
    compute_spatial_vector,
    compute_union_box,
    normalize_name,
)

def inspect_models():
    print("=" * 80)
    print("STEP 1: INSPECTION OF SAVED CHECKPOINTS AND BUFFERS")
    print("=" * 80)
    
    for mname in ["saved_models_v1.0", "saved_models_v2.0"]:
        print(f"\n--- Model Directory: {mname} ---")
        cpath = os.path.join(mname, "vynix_full_cache.pt")
        mpath = os.path.join(mname, "vynix_full_model.pth")
        
        cdata = torch.load(cpath, map_location="cpu", weights_only=True)
        print("Cache Keys in dictionary:")
        for k, v in cdata.items():
            if isinstance(v, torch.Tensor):
                print(f"  cache['{k}']: shape={list(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  cache['{k}']: type={type(v)}, val={v}")
                
        mdata = torch.load(mpath, map_location="cpu", weights_only=True)
        print("Model state_dict (parameters and buffers):")
        for k, v in mdata.items():
            if isinstance(v, torch.Tensor):
                val_repr = f"{v.item():.4f}" if v.numel() == 1 else f"numel={v.numel()}"
                print(f"  param/buffer['{k}']: shape={list(v.shape)}, dtype={v.dtype}, val={val_repr}")


def standalone_step_by_step_scoring(
    f_h: torch.Tensor,               # (512,) normalized
    f_o: torch.Tensor,               # (512,) normalized
    f_u: torch.Tensor,               # (512,) normalized
    s_vec: torch.Tensor,             # (8,)
    s_h: float,
    s_o: float,
    iou_val: float,
    b_dist: float,
    verb: str,
    hoi_id: int,
    t_obj: torch.Tensor,             # (512,)
    text_weights: torch.Tensor,      # (600, 512)
    cache_keys: torch.Tensor,        # (N, 1536)
    cache_values: torch.Tensor,      # (N, 600)
    alpha: float,
    beta: float,
    spatial_mlp: nn.Module,
    sigma: float,
    det_lambda: float,
    obj_gamma: float,
) -> float:
    """
    Computes final HOI score strictly step-by-step from raw inputs without calling adapter.forward.
    """
    # 1. Visual Feature Fusion (L2 normalized concatenation of 3 crops)
    # Each crop feature is unit norm (||f|| = 1.0); concatenated norm is sqrt(3)
    raw_fused = torch.cat([f_h, f_o, f_u], dim=-1)  # (1536,)
    f_fused = raw_fused / raw_fused.norm(dim=-1, keepdim=True)  # (1536,)

    # 2. Zero-Shot CLIP Logits:
    # Code extracts the union slice f_fused[1024:1536] (which has norm 1/sqrt(3) ~= 0.577)
    # and projects against text_weights.
    f_union_slice = f_fused[1024:1536]
    clip_logits = f_union_slice @ text_weights.T  # (600,)

    # 3. Cache Similarity & Affinity
    affinity = f_fused @ cache_keys.T  # (N,)
    affinity = torch.exp(-beta * (1.0 - affinity))  # (N,)
    cache_logits = affinity @ cache_values  # (600,)

    # 4. Residual Blending
    blended_logits = clip_logits + alpha * cache_logits  # (600,)

    # 5. Spatial MLP Prior
    # 8 -> Linear(8,64) -> LayerNorm(64) -> ReLU -> Linear(64,600) -> Sigmoid
    spatial_priors = spatial_mlp(s_vec.unsqueeze(0)).squeeze(0)  # (600,)
    spatial_log_prior = torch.log(spatial_priors + 1e-6)
    final_logits = blended_logits + 0.5 * spatial_log_prior  # (600,)

    # 6. Softmax Normalization
    probs = torch.softmax(final_logits, dim=-1)  # (600,)
    raw_prob = probs[hoi_id].item()

    # 7. Detector Calibration
    det_score = (s_h * s_o) ** det_lambda

    # 8. Soft Geometric Veto Gate
    gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma)

    # 9. Object Semantic Identity Gate
    cos_sim = max(0.0, (f_o @ t_obj).item())
    obj_gate = (cos_sim ** obj_gamma) if obj_gamma > 0 else 1.0

    # 10. Final Multiplicative Score
    final_score = det_score * raw_prob * gate * obj_gate
    return final_score


def verify_pipeline_vs_standalone(dataset_dir: str = r"E:\Dataset"):
    print("\n" + "=" * 80)
    print("STEP 2: PIPELINE VS STANDALONE STEP-BY-STEP VERIFICATION (20 TEST PAIRS)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = HOIMeta(dataset_dir)
    frozen_json_path = "review_response/frozen_hyperparameters.json"
    with open(frozen_json_path, "r", encoding="utf-8") as f:
        hp_configs = json.load(f)

    test_file = os.path.join(dataset_dir, "data", "test-00000-of-00004.parquet")
    pf = pq.ParquetFile(test_file)
    batch = next(pf.iter_batches(batch_size=32))
    df = batch.to_pandas()

    for mname in ["vynix_1.0", "vynix_2.0"]:
        print(f"\n>>> Verifying model: {mname} on 20 candidate pairs...")
        hp = hp_configs[mname]
        sigma = hp["sigma"]
        det_lambda = hp["lambda"]
        obj_gamma = hp["gamma_obj"]
        det_thresh = hp["detector_threshold"]

        mdir = "saved_models_v1.0" if mname == "vynix_1.0" else "saved_models_v2.0"
        det_model = "yolov8m.pt" if mname == "vynix_1.0" else "yolov8x.pt"

        extractor = MultiStreamFeatureExtractor(
            device=device,
            detector_conf=det_thresh,
            use_3stream=True,
            detector_model=det_model,
            clip_model="openai/clip-vit-base-patch16",
        )
        extractor.yolo.to(device).eval()

        text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(device)
        obj_text_weights = extractor.build_object_text_weights(meta)

        cpath = os.path.join(mdir, "vynix_full_cache.pt")
        mpath = os.path.join(mdir, "vynix_full_model.pth")
        cdata = torch.load(cpath, map_location=device, weights_only=True)
        adapter = Vynix3StreamAdapter(
            cache_keys=cdata["cache_keys"],
            cache_values=cdata["cache_values"],
            use_spatial_mlp=True,
            num_classes=meta.num_classes,
        ).to(device).eval()
        adapter.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))

        alpha_val = adapter.alpha.item()
        beta_val = adapter.beta.item()

        pairs_tested = 0
        max_abs_diff = 0.0
        all_diffs = []

        for _, row in df.iterrows():
            if pairs_tested >= 20:
                break
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
                continue

            pairs = [(p, o) for p in persons for o in objects if p[1] >= det_thresh and o[1] >= det_thresh]
            for (p_box, p_conf), (o_box, o_conf, o_cls) in pairs:
                if pairs_tested >= 20:
                    break
                coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
                if not coco_name:
                    continue
                hico_name = normalize_name(coco_name)
                entries = meta.obj_to_entries.get(hico_name, [])
                if not entries:
                    continue

                u_box = compute_union_box(p_box, o_box)
                s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h).to(device)
                iou_val = compute_iou(p_box, o_box)
                b_dist = compute_box_distance(p_box, o_box, img_w, img_h)

                # Pipeline visual feature extraction
                boxes_to_extract = [(p_box, o_box, u_box)]
                f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                s_vecs_batch = s_vec.unsqueeze(0)
                f_unions = f_fused_batch[:, 1024:1536]
                f_objs = f_fused_batch[:, 512:1024]
                f_objs_norm = f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)

                # 1. Pipeline Computation
                with torch.no_grad():
                    clip_logits = f_unions @ text_weights.T
                    final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
                    pipeline_probs = torch.softmax(final_logits, dim=-1).squeeze(0)

                t_obj = obj_text_weights.get(hico_name, torch.zeros(512)).to(device)
                cos_sim = max(0.0, (f_objs_norm[0] @ t_obj).item())
                obj_gate = (cos_sim ** obj_gamma) if obj_gamma > 0 else 1.0
                det_score = (p_conf * o_conf) ** det_lambda

                # Test on first entry of this pair
                verb, _, hoi_id = entries[0]
                gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma)
                pipeline_score = det_score * pipeline_probs[hoi_id].item() * gate * obj_gate

                # 2. Standalone Computation
                # Reconstruct raw normalized crop features (each of norm 1.0)
                sqrt3 = 3.0 ** 0.5
                f_h_raw = f_fused_batch[0, 0:512] * sqrt3
                f_o_raw = f_fused_batch[0, 512:1024] * sqrt3
                f_u_raw = f_fused_batch[0, 1024:1536] * sqrt3

                standalone_score = standalone_step_by_step_scoring(
                    f_h=f_h_raw,
                    f_o=f_o_raw,
                    f_u=f_u_raw,
                    s_vec=s_vec,
                    s_h=p_conf,
                    s_o=o_conf,
                    iou_val=iou_val,
                    b_dist=b_dist,
                    verb=verb,
                    hoi_id=hoi_id,
                    t_obj=t_obj,
                    text_weights=text_weights,
                    cache_keys=adapter.cache_keys,
                    cache_values=adapter.cache_values,
                    alpha=alpha_val,
                    beta=beta_val,
                    spatial_mlp=adapter.spatial_mlp,
                    sigma=sigma,
                    det_lambda=det_lambda,
                    obj_gamma=obj_gamma,
                )

                diff = abs(pipeline_score - standalone_score)
                all_diffs.append(diff)
                max_abs_diff = max(max_abs_diff, diff)
                pairs_tested += 1
                if pairs_tested <= 5 or pairs_tested == 20:
                    print(f"  Pair {pairs_tested:2d}: pipeline={pipeline_score:.8f}, standalone={standalone_score:.8f}, diff={diff:.2e}")

        print(f"  >>> Model {mname}: Tested {pairs_tested} pairs | Max absolute difference = {max_abs_diff:.2e}")
        assert max_abs_diff < 1e-6, f"Mismatch exceeded tolerance: {max_abs_diff}"
        print(f"  >>> PASSED: Standalone computation is bit-exact with pipeline (diff < 1e-6)!")


if __name__ == "__main__":
    inspect_models()
    verify_pipeline_vs_standalone()
