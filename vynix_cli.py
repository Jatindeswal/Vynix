#!/usr/bin/env python3
"""
Project Vynix: Production-Grade Unified CLI
============================================
Provides unified subcommands for Human-Object Interaction (HOI) detection,
evaluation on HICO-DET benchmarks, and few-shot adapter training.

Subcommands:
  - vynix predict --image <path> [--detector yolov8m.pt] [--model-dir saved_models_vitb16]
  - vynix eval --dataset-dir <path> [--limit 100]
  - vynix train --dataset-dir <path> [--epochs 15]
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path


# Package metadata
__version__ = "1.0.0"
__banner__ = r"""
 __      __            _        
 \ \    / /           (_)       
  \ \  / /   _ _ __   ___  __  
   \ \/ / | | | '_ \ | \ \/ /  
    \  /| |_| | | | || |>  <   
     \/  \__, |_| |_||_/_/\_\  
          __/ |                 
         |___/                  
 Project Vynix — Production HOI Detection CLI (v%s)
 [Geometric Hallucination Veto Architecture]
""" % __version__


def get_default_device():
    """Detects available hardware acceleration without crashing."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_list_action_path(explicit_path=None, model_dir=None):
    """Searches for list_action.csv in candidate locations."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if model_dir:
        candidates.append(os.path.join(model_dir, "list_action.csv"))
    
    # Current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "list_action.csv"))
    candidates.append(os.path.join(os.getcwd(), "list_action.csv"))
    
    # Common dataset directories
    candidates.append(r"E:\Dataset\list_action.csv")
    candidates.append(r"D:\Dataset\list_action.csv")
    candidates.append("/app/list_action.csv")

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def run_predict(args):
    """Executes single-image HOI inference with geometric veto logic."""
    print("=" * 78)
    print("  PROJECT VYNIX — HOI DETECTION PREDICTION PIPELINE")
    print("=" * 78)

    if not os.path.isfile(args.image):
        print(f"[ERROR] Specified image does not exist: {args.image}")
        sys.exit(1)

    # Resolve device
    device = args.device or get_default_device()
    print(f"  • Target Image   : {args.image}")
    print(f"  • Detector       : {args.detector}")
    print(f"  • Model Dir      : {args.model_dir}")
    print(f"  • Hardware Device: {device}")

    # Resolve list_action.csv
    list_action_csv = find_list_action_path(args.list_action, args.model_dir)
    if not list_action_csv:
        print("[ERROR] Could not locate 'list_action.csv'. Please provide --list-action <path>.")
        sys.exit(1)
    dataset_meta_dir = os.path.dirname(list_action_csv)

    # Determine CLIP backbone model
    clip_model = args.clip_model
    if not clip_model:
        eval_json = os.path.join(args.model_dir, "eval_results.json")
        if os.path.isfile(eval_json):
            try:
                with open(eval_json, "r") as f:
                    ed = json.load(f)
                    clip_model = ed.get("clip_model")
            except Exception:
                pass
        if not clip_model:
            if "vitb16" in args.model_dir.lower():
                clip_model = "openai/clip-vit-base-patch16"
            else:
                clip_model = "openai/clip-vit-base-patch32"

    print(f"  • CLIP Backbone  : {clip_model}")
    print(f"  • Meta Definition: {list_action_csv}\n")

    # Import heavy scientific dependencies lazily
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from vynix_fewshot_adapter import (
        HOIMeta,
        MultiStreamFeatureExtractor,
        Vynix3StreamAdapter,
        compute_iou,
        compute_union_box,
        compute_spatial_vector,
        compute_box_distance,
        compute_soft_geometric_gate,
        normalize_name,
        COCO_CLASSES,
        CONTACT_VERBS
    )

    print("  [1/4] Loading HOI action taxonomy and feature extractors...")
    meta = HOIMeta(dataset_meta_dir)
    extractor = MultiStreamFeatureExtractor(
        device=device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector,
        clip_model=clip_model
    )
    text_weights = extractor.build_text_weights(meta).to(device)

    # Check for trained adapter model and cache
    cache_path = os.path.join(args.model_dir, "vynix_full_cache.pt")
    model_path = os.path.join(args.model_dir, "vynix_full_model.pth")
    adapter = None

    if os.path.isfile(cache_path) and os.path.isfile(model_path):
        print(f"  [2/4] Loading trained 3-Stream Adapter weights from {args.model_dir}...")
        cache_data = torch.load(cache_path, map_location=device, weights_only=True)
        cache_keys = cache_data["cache_keys"]
        cache_values = cache_data["cache_values"]

        adapter = Vynix3StreamAdapter(
            cache_keys=cache_keys,
            cache_values=cache_values,
            use_spatial_mlp=True,
            num_classes=meta.num_classes
        )
        adapter.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        adapter.to(device).eval()
    else:
        print("  [2/4] Note: Model weights not found in model-dir. Running in Zero-Shot mode.")

    print(f"  [3/4] Performing node detection on {args.image}...")
    pil_img = Image.open(args.image).convert("RGB")
    img_w, img_h = pil_img.size

    persons, objects = extractor.detect(pil_img)
    print(f"        Detected {len(persons)} human candidate(s) and {len(objects)} object candidate(s).")

    if not persons or not objects:
        print("\n  [RESULT] No interactive human-object pairs detected in the image.")
        return

    # Form interactive pairs
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
            "p_box": p_box, "p_conf": p_conf,
            "o_box": o_box, "o_conf": o_conf, "o_cls": o_cls, "o_name": hico_name,
            "u_box": u_box, "s_vec": s_vec, "iou": iou_val, "dist": b_dist,
            "entries": entries
        })

    if not valid_pairs:
        print("\n  [RESULT] Detected objects do not match any HICO-DET interaction classes.")
        return

    print("  [4/4] Computing visual-spatial alignment and geometric veto gate...")
    boxes_to_extract = [(vp["p_box"], vp["o_box"], vp["u_box"]) for vp in valid_pairs]
    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(device)
    s_vecs_batch = torch.stack([vp["s_vec"] for vp in valid_pairs]).to(device)
    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch

    with torch.no_grad():
        clip_logits = f_unions @ text_weights.T
        if adapter is not None:
            final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
            probs_batch = torch.softmax(final_logits, dim=1).cpu()
        else:
            probs_batch = torch.softmax(clip_logits * 100.0, dim=1).cpu()

    predictions = []
    total_vetoes = 0

    for k_idx, vp in enumerate(valid_pairs):
        p_box = vp["p_box"]
        o_box = vp["o_box"]
        o_name = vp["o_name"]
        iou_val = vp["iou"]
        b_dist = vp["dist"]
        p_conf = vp["p_conf"]
        o_conf = vp["o_conf"]

        for verb, gerund, hoi_id in vp["entries"]:
            raw_prob = probs_batch[k_idx, hoi_id].item()
            gate = compute_soft_geometric_gate(verb, iou_val, b_dist)
            is_vetoed = False

            if gate < 0.05 or (verb in CONTACT_VERBS and iou_val == 0.0):
                is_vetoed = True
                total_vetoes += 1

            conf = p_conf * o_conf * raw_prob * gate
            interaction_label = f"person {verb} {o_name}"
            predictions.append({
                "hoi_id": hoi_id,
                "label": interaction_label,
                "verb": verb,
                "object": o_name,
                "confidence": conf,
                "raw_prob": raw_prob,
                "gate": gate,
                "is_vetoed": is_vetoed,
                "iou": iou_val,
                "dist": b_dist,
                "p_box": p_box,
                "o_box": o_box,
                "p_conf": p_conf,
                "o_conf": o_conf,
            })

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_preds = [p for p in predictions if p["confidence"] >= args.conf][:args.top_k]
    if not top_preds and predictions:
        top_preds = predictions[:args.top_k]

    print("\n" + "=" * 78)
    print("  🏆 TOP PREDICTED HUMAN-OBJECT INTERACTIONS")
    print("=" * 78)
    print(f"  {'Rank':<5} | {'Interaction':<28} | {'Score':<8} | {'IoU':<6} | {'Gate Verdict':<24}")
    print("-" * 78)
    for rank, p in enumerate(top_preds, 1):
        if p["is_vetoed"]:
            verdict = "🛡️ VETOED (Zero-IoU)"
        else:
            verdict = f"✓ APPROVED (Gate={p['gate']:.2f})"
        print(f"  {rank:<5} | {p['label']:<28} | {p['confidence']:>6.2%} | {p['iou']:>5.2f} | {verdict:<24}")
    print("-" * 78)
    print(f"  Total spatial hallucinations vetoed: {total_vetoes}\n")

    # Optional visualization saving
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        draw = ImageDraw.Draw(pil_img)

        # Draw top prediction
        if top_preds:
            best = top_preds[0]
            # Human box (Green)
            draw.rectangle(best["p_box"], outline="#00FF00", width=3)
            # Object box (Cyan)
            draw.rectangle(best["o_box"], outline="#00FFFF", width=3)
            # Text tag
            tag = f"{best['label']} ({best['confidence']:.1%})"
            draw.text((best["p_box"][0], max(0, best["p_box"][1] - 18)), tag, fill="#00FF00")

        pil_img.save(str(out_path))
        print(f"  ✓ Annotated visualization saved to: {out_path.resolve()}")


def run_eval(args):
    """Executes HICO-DET benchmark evaluation."""
    print("=" * 78)
    print("  PROJECT VYNIX — HICO-DET BENCHMARK EVALUATION")
    print("=" * 78)

    if not os.path.isdir(args.dataset_dir):
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    import glob
    import torch
    from vynix_fewshot_adapter import (
        HOIMeta,
        MultiStreamFeatureExtractor,
        Vynix3StreamAdapter,
    )
    from test_vynix_pretrained import evaluate_pretrained

    device = args.device or get_default_device()
    detector = args.detector
    model_dir = args.model_dir

    # Auto-detect clip model
    clip_model = args.clip_model
    if not clip_model:
        eval_json = os.path.join(model_dir, "eval_results.json")
        if os.path.isfile(eval_json):
            try:
                with open(eval_json, "r") as f:
                    ed = json.load(f)
                    clip_model = ed.get("clip_model")
            except Exception:
                pass
        if not clip_model:
            clip_model = "openai/clip-vit-base-patch16" if "vitb16" in model_dir.lower() else "openai/clip-vit-base-patch32"

    print(f"  • Dataset Directory : {args.dataset_dir}")
    print(f"  • Model Directory   : {model_dir}")
    print(f"  • Detector Model    : {detector}")
    print(f"  • CLIP Backbone     : {clip_model}")
    print(f"  • Evaluation Limit  : {args.limit if args.limit and args.limit > 0 else 'Full Test Set'}")
    print(f"  • Computation Device: {device}\n")

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=detector,
        clip_model=clip_model
    )
    text_weights = extractor.build_text_weights(meta)

    cache_path = os.path.join(model_dir, "vynix_full_cache.pt")
    model_path = os.path.join(model_dir, "vynix_full_model.pth")

    if not os.path.isfile(cache_path) or not os.path.isfile(model_path):
        print(f"[ERROR] Trained weights or cache missing in {model_dir}.")
        print("        Run 'vynix train' first or specify a valid --model-dir.")
        sys.exit(1)

    print(f"  Loading cache: {cache_path}")
    cache_data = torch.load(cache_path, map_location=device, weights_only=True)
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_data["cache_keys"],
        cache_values=cache_data["cache_values"],
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    )
    print(f"  Loading model weights: {model_path}")
    adapter.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))
    if not test_files:
        print(f"[ERROR] No test-*.parquet shards found in {args.dataset_dir}/data")
        sys.exit(1)

    limit = args.limit if args.limit and args.limit > 0 else None
    res = evaluate_pretrained(adapter, extractor, text_weights, meta, test_files, device, eval_limit=limit)

    print("\n" + "=" * 78)
    print("  🏆 FINAL BENCHMARK EVALUATION RESULTS")
    print("=" * 78)
    print(f"  Evaluated Images     : {res['n_images']}")
    print(f"  Full mAP             : {res['mAP_full']:>6.2f}%")
    print(f"  Rare mAP (<10 train) : {res['mAP_rare']:>6.2f}%")
    print(f"  Non-Rare mAP (>=10)  : {res['mAP_non_rare']:>6.2f}%")
    print(f"  Hallucinations Vetoed: {res['vetoes']}")
    print("=" * 78)

    if args.output_json:
        out_json = args.output_json
    elif limit is None:
        out_json = os.path.join(model_dir, "eval_results.json")
    else:
        out_json = os.path.join(model_dir, f"eval_results_limit{limit}.json")

    with open(out_json, "w") as fp:
        json.dump({
            "mAP_full": res["mAP_full"],
            "mAP_rare": res["mAP_rare"],
            "mAP_non_rare": res["mAP_non_rare"],
            "vetoes": res["vetoes"],
            "evaluated_images": res["n_images"],
            "detector": detector,
            "clip_model": clip_model
        }, fp, indent=2)
    print(f"  ✓ Evaluation metrics saved to {out_json}")


def run_train(args):
    """Executes full dataset feature caching and adapter training."""
    print("=" * 78)
    print("  PROJECT VYNIX — FULL DATASET TRAINING PIPELINE")
    print("=" * 78)

    if not os.path.isdir(args.dataset_dir):
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    import glob
    import torch
    from vynix_fewshot_adapter import (
        HOIMeta,
        MultiStreamFeatureExtractor,
        Vynix3StreamAdapter,
    )
    from train_vynix_full import build_full_dataset_cache, train_adapter_full

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or get_default_device()

    print(f"  • Dataset Directory : {args.dataset_dir}")
    print(f"  • Output Directory  : {args.output_dir}")
    print(f"  • Epochs            : {args.epochs}")
    print(f"  • Batch Size        : {args.batch_size}")
    print(f"  • Detector Model    : {args.detector}")
    print(f"  • CLIP Backbone     : {args.clip_model}")
    print(f"  • Hardware Device   : {device}\n")

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector,
        clip_model=args.clip_model
    )
    text_weights = extractor.build_text_weights(meta)

    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))
    if not train_files:
        print(f"[ERROR] No train-*.parquet shards found in {args.dataset_dir}/data")
        sys.exit(1)

    max_shots = args.max_shots if args.max_shots > 0 else None
    cache_keys, cache_values, cache_spatials = build_full_dataset_cache(
        train_files, meta, extractor, device,
        output_dir=args.output_dir,
        max_images=args.limit,
        max_shots_per_class=max_shots
    )

    cache_path = os.path.join(args.output_dir, "vynix_full_cache.pt")
    torch.save({
        "cache_keys": cache_keys,
        "cache_values": cache_values,
        "cache_spatials": cache_spatials
    }, cache_path)
    print(f"  ✓ Saved cache tensor to {cache_path}")

    adapter = Vynix3StreamAdapter(
        cache_keys=cache_keys.to(device),
        cache_values=cache_values.to(device),
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    )

    trained_adapter = train_adapter_full(
        adapter=adapter,
        train_features=cache_keys,
        train_labels=cache_values,
        train_spatials=cache_spatials,
        text_weights=text_weights,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size
    )

    model_path = os.path.join(args.output_dir, "vynix_full_model.pth")
    torch.save(trained_adapter.state_dict(), model_path)
    print(f"\n  ✓ Saved trained adapter model to {model_path}")


def build_parser():
    """Builds and returns the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="vynix",
        description="Project Vynix: Production CLI for Few-Shot HOI Detection with Geometric Hallucination Veto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vynix predict --image demo.jpg --detector yolov8m.pt --model-dir saved_models_vitb16
  vynix eval --dataset-dir /data/hico_det --limit 100
  vynix train --dataset-dir /data/hico_det --epochs 15
        """
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"Project Vynix CLI v{__version__}"
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="subcommands",
        description="Valid Vynix actions",
        required=False
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Subcommand: predict
    # ──────────────────────────────────────────────────────────────────────────
    p_predict = subparsers.add_parser(
        "predict",
        help="Run HOI detection and geometric hallucination veto on an input image",
        description="Run inference on a single image, predicting human-object interactions and applying the geometric veto gate."
    )
    p_predict.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (JPEG or PNG)"
    )
    p_predict.add_argument(
        "--detector",
        type=str,
        default="yolov8m.pt",
        help="YOLO detector model weight file (default: yolov8m.pt)"
    )
    p_predict.add_argument(
        "--model-dir",
        type=str,
        default="saved_models_vitb16",
        help="Directory containing trained model weights and cache (default: saved_models_vitb16)"
    )
    p_predict.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="CLIP backbone (auto-detected if None, e.g. openai/clip-vit-base-patch16)"
    )
    p_predict.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda' or 'cpu', auto-detected by default)"
    )
    p_predict.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold for displaying interaction predictions (default: 0.10)"
    )
    p_predict.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of top interaction predictions to display (default: 5)"
    )
    p_predict.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save annotated visualization image"
    )
    p_predict.add_argument(
        "--list-action",
        type=str,
        default=None,
        help="Optional path to list_action.csv metadata file"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Subcommand: eval
    # ──────────────────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser(
        "eval",
        help="Evaluate model performance on the HICO-DET benchmark",
        description="Evaluate Vynix on HICO-DET test shards, calculating Full, Rare, and Non-Rare mAP."
    )
    p_eval.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing list_action.csv and data/test-*.parquet"
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of test images for evaluation (default: 100, 0 for full test set)"
    )
    p_eval.add_argument(
        "--model-dir",
        type=str,
        default="saved_models_vitb16",
        help="Directory containing trained model weights and cache (default: saved_models_vitb16)"
    )
    p_eval.add_argument(
        "--detector",
        type=str,
        default="yolov8m.pt",
        help="YOLO detector model weight file (default: yolov8m.pt)"
    )
    p_eval.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="CLIP backbone (auto-detected if None, e.g. openai/clip-vit-base-patch16)"
    )
    p_eval.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on ('cuda' or 'cpu')"
    )
    p_eval.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Subcommand: train
    # ──────────────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser(
        "train",
        help="Train the Vynix 3-Stream Adapter on HICO-DET training shards",
        description="Extract multi-stream visual and geometric caches, then train the Tip-Adapter and Spatial MLP."
    )
    p_train.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing list_action.csv and data/train-*.parquet"
    )
    p_train.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)"
    )
    p_train.add_argument(
        "--output-dir",
        type=str,
        default="saved_models_vitb16",
        help="Output directory to save model weights and cache (default: saved_models_vitb16)"
    )
    p_train.add_argument(
        "--detector",
        type=str,
        default="yolov8m.pt",
        help="YOLO detector model weight file (default: yolov8m.pt)"
    )
    p_train.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP vision backbone (default: openai/clip-vit-base-patch16)"
    )
    p_train.add_argument(
        "--max-shots",
        type=int,
        default=50,
        help="Max exemplars per HOI class in the cache (default: 50)"
    )
    p_train.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit training images for testing"
    )
    p_train.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on ('cuda' or 'cpu')"
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        print(__banner__)
        parser.print_help()
        sys.exit(0)

    try:
        if args.subcommand == "predict":
            run_predict(args)
        elif args.subcommand == "eval":
            run_eval(args)
        elif args.subcommand == "train":
            run_train(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n[INFO] Operation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
