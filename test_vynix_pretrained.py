import os
import argparse
import glob
import torch
from tqdm import tqdm
import pandas as pd
import io
from PIL import Image

# Import existing core components
from vynix_fewshot_adapter import (
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    CONTACT_VERBS,
    compute_iou,
    compute_union_box,
    compute_spatial_vector,
    compute_box_distance,
    compute_soft_geometric_gate,
    normalize_name,
    COCO_CLASSES,
    parse_int_list,
    compute_ap
)
import numpy as np

def evaluate_pretrained(adapter, extractor, text_weights, meta, test_files, device, eval_limit=None, det_lambda=0.8, adaptive_alpha=True, obj_text_weights=None, obj_gamma=0.15):
    import pyarrow.parquet as pq
    import gc

    extractor.yolo.to(device).eval()
    adapter.to(device).eval()
    text_weights_dev = text_weights.to(device)

    # Class-adaptive alpha weighting: higher for Rare classes (<10 shots), lower for Non-Rare
    rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    if adaptive_alpha:
        alpha_vec = torch.tensor([0.55 if h in rare_ids else 0.25 for h in range(meta.num_classes)], device=device)
    else:
        alpha_vec = None

    total_available = sum(pq.read_metadata(f).num_rows for f in test_files)
    n_target = min(total_available, eval_limit) if eval_limit else total_available
    print(f"\n  [Evaluation] Indexed {total_available} test images across {len(test_files)} shards. Evaluating: {n_target} images...")
    print(f"  [Optimization] Score Calibration lambda={det_lambda:.2f} | Adaptive Alpha: {adaptive_alpha}")

    all_preds = {}
    all_gt = {}
    total_vetoes = 0
    processed_count = 0

    with tqdm(total=n_target, desc="  Evaluating", unit="img") as pbar:
        for f in test_files:
            pf = pq.ParquetFile(f)
            reached_limit = False
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if eval_limit and processed_count >= eval_limit:
                        reached_limit = True
                        break
                    img_idx = processed_count
                    processed_count += 1
                    pbar.update(1)

                    all_gt[img_idx] = set(parse_int_list(str(row.get("positive_objects", ""))))

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
                        all_preds[img_idx] = {}
                        continue

                    img_w, img_h = pil_img.size
                    preds = {}

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
                        valid_pairs.append((p_box, p_conf, o_box, o_conf, u_box, s_vec, iou_val, b_dist, entries))

                    if not valid_pairs:
                        all_preds[img_idx] = {}
                        continue

                    boxes_to_extract = [(p, o, u) for (p, _, o, _, u, _, _, _, _) in valid_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
                    s_vecs_batch = torch.stack([s for (_, _, _, _, _, s, _, _, _) in valid_pairs]).to(device)

                    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch
                    with torch.no_grad():
                        clip_logits = f_unions @ text_weights_dev.T
                        final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch, alpha_override=alpha_vec)
                        probs_batch = torch.softmax(final_logits, dim=1).cpu()

                    if obj_text_weights is not None and obj_gamma > 0.0 and f_fused_batch.shape[-1] == 1536:
                        f_objs = f_fused_batch[:, 512:1024]
                        f_objs_norm = f_objs / (f_objs.norm(dim=-1, keepdim=True) + 1e-6)
                    else:
                        f_objs_norm = None

                    for k_idx, (_, p_conf, _, o_conf, _, _, iou_val, b_dist, entries) in enumerate(valid_pairs):
                        det_score = (p_conf * o_conf) ** det_lambda
                        obj_gate = 1.0
                        if f_objs_norm is not None and entries:
                            hico_name = meta.hoi_to_obj[entries[0][2]]
                            if hico_name in obj_text_weights:
                                t_obj = obj_text_weights[hico_name]
                                cos_sim = (f_objs_norm[k_idx] @ t_obj).item()
                                obj_gate = max(0.0, cos_sim) ** obj_gamma

                        for verb, _, hoi_id in entries:
                            raw_prob = probs_batch[k_idx, hoi_id].item()

                            # Vynix Soft Geometric Gate
                            gate = compute_soft_geometric_gate(verb, iou_val, b_dist)
                            if gate < 0.05:
                                total_vetoes += 1

                            conf = det_score * raw_prob * gate * obj_gate
                            if hoi_id not in preds or conf > preds[hoi_id]:
                                preds[hoi_id] = conf

                    all_preds[img_idx] = preds
                    try:
                        del pil_img
                    except NameError:
                        pass

                del df
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if reached_limit:
                    break
            if reached_limit:
                break

    all_hoi_ids = sorted(meta.hoi_to_obj.keys())
    per_class_ap = {}
    for hoi_id in all_hoi_ids:
        scores = [all_preds.get(i, {}).get(hoi_id, 0.0) for i in range(processed_count)]
        labels = [1 if hoi_id in all_gt.get(i, set()) else 0 for i in range(processed_count)]
        per_class_ap[hoi_id] = compute_ap(scores, labels)

    full_aps = [per_class_ap[h] for h in all_hoi_ids]
    rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0} # Using identical pseudo-rare split logic from vynix_fewshot_adapter for consistency
    nonrare_ids = set(range(meta.num_classes)) - rare_ids
    rare_aps = [per_class_ap[h] for h in rare_ids if h in per_class_ap]
    nonrare_aps = [per_class_ap[h] for h in nonrare_ids if h in per_class_ap]

    return {
        "mAP_full": float(np.mean(full_aps)) * 100,
        "mAP_rare": float(np.mean(rare_aps)) * 100 if rare_aps else 0.0,
        "mAP_non_rare": float(np.mean(nonrare_aps)) * 100 if nonrare_aps else 0.0,
        "vetoes": total_vetoes,
        "n_images": processed_count,
        "per_class_ap": per_class_ap,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Project Vynix Pretrained Model")
    parser.add_argument("--dataset-dir", type=str, default="/app")
    parser.add_argument("--model-dir", type=str, default="saved_models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit test images for testing")
    parser.add_argument("--detector-model", type=str, default="yolov8m.pt", help="YOLO detector model (e.g. yolov8m.pt, yolov8l.pt, yolov8n.pt)")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16", help="CLIP vision backbone (e.g. openai/clip-vit-base-patch16, openai/clip-vit-base-patch32)")
    parser.add_argument("--det-lambda", type=float, default=0.8, help="Sublinear detector score calibration exponent (default: 0.8)")
    parser.add_argument("--no-prompt-ensemble", action="store_true", help="Disable multi-template prompt ensemble")
    parser.add_argument("--no-adaptive-alpha", action="store_true", help="Disable class-adaptive alpha blending")
    parser.add_argument("--obj-gamma", type=float, default=0.15, help="Exponent for isolated object semantic verification gate (default: 0.15, 0 to disable)")
    parser.add_argument("--no-obj-gate", action="store_true", help="Disable isolated object semantic verification gate")
    args = parser.parse_args()

    print("===========================================================================")
    print("  PROJECT VYNIX — PRETRAINED MODEL EVALUATION")
    print("===========================================================================")
    print(f"  Device: {args.device}")
    print(f"  Detector: {args.detector_model}")
    print(f"  CLIP Backbone: {args.clip_model}")
    print(f"  Score Calibration lambda: {args.det_lambda}")
    print(f"  Prompt Ensemble: {not args.no_prompt_ensemble}")
    print(f"  Adaptive Alpha: {not args.no_adaptive_alpha}")
    print(f"  Object Gate (gamma): {args.obj_gamma if not args.no_obj_gate else 0.0}")

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=args.device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector_model,
        clip_model=args.clip_model
    )
    use_ensemble = not args.no_prompt_ensemble
    text_weights = extractor.build_text_weights(meta, use_ensemble=use_ensemble)

    # 1. Load the Cache
    cache_path = os.path.join(args.model_dir, "vynix_full_cache.pt")
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}. Run train_vynix_full.py first.")
        return

    print(f"\n  Loading cache from {cache_path}...")
    cache_data = torch.load(cache_path, map_location=args.device)
    cache_keys = cache_data["cache_keys"]
    cache_values = cache_data["cache_values"]

    # 2. Instantiate and Load the Model
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_keys,
        cache_values=cache_values,
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    )

    model_path = os.path.join(args.model_dir, "vynix_full_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Run train_vynix_full.py first.")
        return

    print(f"  Loading model weights from {model_path}...")
    adapter.load_state_dict(torch.load(model_path, map_location=args.device))

    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))

    if not args.no_obj_gate and args.obj_gamma > 0.0:
        print("  Precomputing multi-template object text embeddings for verification gate...")
        obj_text_weights = extractor.build_object_text_weights(meta)
    else:
        obj_text_weights = None

    # 3. Evaluate
    res = evaluate_pretrained(
        adapter=adapter,
        extractor=extractor,
        text_weights=text_weights,
        meta=meta,
        test_files=test_files,
        device=args.device,
        eval_limit=args.limit,
        det_lambda=args.det_lambda,
        adaptive_alpha=not args.no_adaptive_alpha,
        obj_text_weights=obj_text_weights,
        obj_gamma=args.obj_gamma
    )

    print("\n===========================================================================")
    print("  🏆 FINAL PRETRAINED MODEL EVALUATION RESULTS")
    print("===========================================================================")
    print(f"  Full mAP       : {res['mAP_full']:>6.2f}%")
    print(f"  Rare mAP       : {res['mAP_rare']:>6.2f}%")
    print(f"  Non-Rare mAP   : {res['mAP_non_rare']:>6.2f}%")
    print(f"  Vetoes Triggered: {res['vetoes']}")
    print("===========================================================================")

    if args.limit is None:
        import json
        out_json = os.path.join(args.model_dir, "eval_results.json")
        with open(out_json, "w") as fp:
            json.dump({
                "mAP_full": res["mAP_full"],
                "mAP_rare": res["mAP_rare"],
                "mAP_non_rare": res["mAP_non_rare"],
                "vetoes": res["vetoes"],
                "detector": args.detector_model,
                "clip_model": args.clip_model,
                "det_lambda": args.det_lambda,
                "obj_gamma": args.obj_gamma if not args.no_obj_gate else 0.0,
                "total_test_images": res["n_images"]
            }, fp, indent=2)
        print(f"  ✓ Saved full evaluation results to {out_json}")
    else:
        print(f"  (Evaluation run with limit={args.limit}; eval_results.json preserved)")

if __name__ == "__main__":
    main()
