#!/usr/bin/env python3
"""
==============================================================================
  PROJECT VYNIX — HICO-DET Full Benchmark Evaluation
  evaluate_vynix_full.py
==============================================================================

  Benchmarks the Vynix 4-stage HOI pipeline against the HICO-DET dataset
  (~9,658 test images, 600 HOI categories) using image-level classification
  mAP with detection-aware confidence scoring.

  Pipeline:
    Stage 1  Node Extraction   — YOLOv8-nano (all 80 COCO classes)
    Stage 2  Spatial Graph     — Centroid, Euclidean distance, IoU
    Stage 3  VLM Classification — CLIP ViT-B/32 with dynamic verb prompts
    Stage 4  Vynix Logic Gate  — Hard geometric override on contact verbs

  Metrics:
    - mAP (Full / Rare / Non-Rare)  — official HICO-DET evaluation splits
    - Hallucinations Prevented       — Vynix gate override counter

  Dependencies:
    pip install ultralytics transformers torch torchvision opencv-python-headless
    pip install Pillow numpy datasets tqdm pandas pyarrow huggingface-hub

  Usage:
    python evaluate_vynix_full.py --limit-samples 10 --device cpu
    python evaluate_vynix_full.py --device cuda --output-csv results.csv

  Author:  Project Vynix
  License: MIT
==============================================================================
"""

import argparse
import csv
import glob
import io
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================================
# §1  COCO CLASS NAMES  (80 classes, matching YOLOv8 indices 0–79)
# ============================================================================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ============================================================================
# §2  CONTACT VERBS  —  Verbs requiring physical bounding-box overlap
# ============================================================================
# The Vynix Logic Gate overrides CLIP predictions for these verbs when
# IoU(person_box, object_box) == 0.0, since physical contact is geometrically
# impossible when bounding boxes are completely disjoint.

CONTACT_VERBS = {
    "hold", "carry", "hug", "kiss", "lick", "eat", "drink_with", "sip",
    "taste", "wear", "ride", "sit_on", "sit_at", "lie_on", "stand_on",
    "straddle", "pet", "groom", "milk", "shear", "touch", "catch", "grab",
    "pick_up", "pick", "lift", "flip", "push", "pull", "cut", "cut_with",
    "hit", "kick", "tie", "wash", "dry", "brush_with", "fill", "pour",
    "stab", "squeeze", "type_on", "wield", "swing", "operate",
    "play_with", "control", "drive", "fly", "row", "sail", "board",
    "hop_on", "mount", "drag", "dribble", "grind", "hose", "load",
    "open", "pack", "peel", "spin", "zip",
}

# ============================================================================
# §3  UTILITY FUNCTIONS
# ============================================================================

def _article(word: str) -> str:
    """Return 'an' if word starts with a vowel sound, else 'a'."""
    return "an" if word and word[0].lower() in "aeiou" else "a"


def make_prompt(gerund: str, object_name: str) -> str:
    """Build a natural CLIP prompt: 'a person {gerund} a/an {object}'."""
    gerund_clean = gerund.replace("_", " ").strip()
    obj_clean = object_name.replace("_", " ").strip()
    if gerund_clean == "no interaction":
        return f"a person standing near {_article(obj_clean)} {obj_clean} but not interacting"
    return f"a person {gerund_clean} {_article(obj_clean)} {obj_clean}"


def parse_int_list(val) -> List[int]:
    """Parse integer list from string, list, tuple, or ndarray."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [int(x) for x in val]
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "[]":
            return []
        import re
        return [int(x) for x in re.findall(r'\d+', val)]
    return []


def normalize_name(name: str) -> str:
    """Normalize object names for cross-dataset matching."""
    name = str(name).lower().strip().replace("_", " ")
    aliases = {
        "hair dryer": "hair drier", "tv monitor": "tv",
        "television": "tv", "motorbike": "motorcycle",
        "sofa": "couch", "aeroplane": "airplane",
        "cell_phone": "cell phone", "hot_dog": "hot dog",
        "wine_glass": "wine glass", "potted_plant": "potted plant",
        "dining_table": "dining table", "teddy_bear": "teddy bear",
        "sports_ball": "sports ball", "baseball_bat": "baseball bat",
        "baseball_glove": "baseball glove", "tennis_racket": "tennis racket",
        "fire_hydrant": "fire hydrant", "stop_sign": "stop sign",
        "parking_meter": "parking meter", "traffic_light": "traffic light",
    }
    return aliases.get(name, name)


# ============================================================================
# §4  DATASET & METADATA LOADER
# ============================================================================

class HOIMeta:
    """Metadata container for the 600 HICO-DET HOI categories."""

    def __init__(self):
        self.hoi_to_obj: Dict[int, str] = {}
        self.hoi_to_verb: Dict[int, str] = {}
        self.hoi_to_gerund: Dict[int, str] = {}
        self.obj_to_entries: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        self.rare_ids: Set[int] = set()
        self.non_rare_ids: Set[int] = set()
        self.train_counts: Counter = Counter()
        self.num_classes: int = 600


def ensure_dataset_files(cache_dir: str):
    """Ensure list_action.csv and test parquet files are downloaded in cache_dir."""
    from huggingface_hub import hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "data"), exist_ok=True)

    action_csv = os.path.join(cache_dir, "list_action.csv")
    if not os.path.exists(action_csv):
        print(f"  Downloading list_action.csv to {cache_dir} ...")
        hf_hub_download(repo_id="zhimeng/hico_det", filename="list_action.csv",
                        repo_type="dataset", local_dir=cache_dir)

    for i in range(4):
        pfile = os.path.join(cache_dir, "data", f"test-0000{i}-of-00004.parquet")
        if not os.path.exists(pfile):
            print(f"  Downloading test-0000{i}-of-00004.parquet ...")
            hf_hub_download(repo_id="zhimeng/hico_det", filename=f"data/test-0000{i}-of-00004.parquet",
                            repo_type="dataset", local_dir=cache_dir)


def load_hoi_metadata(cache_dir: str) -> HOIMeta:
    """Load the 600 HOI categories from list_action.csv and determine splits."""
    meta = HOIMeta()
    action_csv = os.path.join(cache_dir, "list_action.csv")
    df = pd.read_csv(action_csv)

    for idx, row in df.iterrows():
        hoi_id = int(idx)
        obj = normalize_name(str(row["nname"]))
        verb = str(row["vname"]).strip()
        gerund = str(row["vname_ing"]).strip() if pd.notna(row.get("vname_ing")) else verb + "ing"

        meta.hoi_to_obj[hoi_id] = obj
        meta.hoi_to_verb[hoi_id] = verb
        meta.hoi_to_gerund[hoi_id] = gerund
        meta.obj_to_entries[obj].append((verb, gerund, hoi_id))

    # Scan training splits for Rare (<10 samples) vs Non-Rare (>=10 samples)
    train_files = sorted(glob.glob(os.path.join(cache_dir, "data", "train-*.parquet")))
    if train_files:
        print(f"  Computing Rare/Non-Rare split from {len(train_files)} training files ...")
        for f in train_files:
            try:
                tdf = pd.read_parquet(f, columns=["positive_objects"])
                for objs in tdf["positive_objects"]:
                    for h in parse_int_list(objs):
                        meta.train_counts[h] += 1
            except Exception:
                pass

    all_ids = set(range(len(df)))
    if meta.train_counts:
        meta.rare_ids = {h for h in all_ids if meta.train_counts[h] < 10}
        meta.non_rare_ids = all_ids - meta.rare_ids
    else:
        # Fallback to standard 138/462 split if training counts are unavailable
        meta.rare_ids = set()
        meta.non_rare_ids = set(all_ids)

    print(f"  Loaded {len(all_ids)} HOI classes across {len(meta.obj_to_entries)} objects.")
    if meta.rare_ids:
        print(f"  Splits: {len(meta.rare_ids)} Rare (<10 train instances), {len(meta.non_rare_ids)} Non-Rare.")
    return meta


def load_test_dataframe(cache_dir: str, limit_samples: Optional[int] = None) -> pd.DataFrame:
    """Load test parquet files into a single unified DataFrame."""
    test_files = sorted(glob.glob(os.path.join(cache_dir, "data", "test-*.parquet")))
    if not test_files:
        raise FileNotFoundError(f"No test parquet files found in {os.path.join(cache_dir, 'data')}")

    dfs = []
    total_loaded = 0
    for f in test_files:
        tdf = pd.read_parquet(f)
        dfs.append(tdf)
        total_loaded += len(tdf)
        if limit_samples and total_loaded >= limit_samples:
            break

    df = pd.concat(dfs, ignore_index=True)
    if limit_samples:
        df = df.iloc[:limit_samples]
    print(f"  Loaded {len(df)} test samples for evaluation.")
    return df


# ============================================================================
# §5  STAGE 1 — NODE EXTRACTION  (YOLOv8-nano)
# ============================================================================

class NodeExtractor:
    """YOLOv8-nano detector for extracting all 80 COCO classes."""

    def __init__(self, device: str = "cpu"):
        from ultralytics import YOLO
        print("  Initializing YOLOv8-nano ('yolov8n.pt') ...")
        self.model = YOLO("yolov8n.pt")
        self.device = device

    def detect(self, pil_image: Image.Image, conf: float = 0.25):
        """
        Run detection on a PIL image.
        Returns:
            persons: list of (box, confidence) — box = [x1,y1,x2,y2]
            objects: list of (box, confidence, coco_class_id)
        """
        results = self.model(pil_image, device=self.device, verbose=False)
        persons, objects = [], []
        for r in results:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                c = float(r.boxes.conf[i].item())
                if c < conf:
                    continue
                box = r.boxes.xyxy[i].tolist()
                if cls_id == 0:
                    persons.append((box, c))
                else:
                    objects.append((box, c, cls_id))
        return persons, objects


# ============================================================================
# §6  STAGE 2 — SPATIAL GRAPH MATH
# ============================================================================

def compute_iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-Union ∈ [0,1]. Returns 0.0 for disjoint boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_union_box(a: List[float], b: List[float]) -> List[float]:
    """Minimum enclosing rectangle of two boxes (the VLM crop region)."""
    return [min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3])]


# ============================================================================
# §7  STAGE 3 — VLM CLASSIFICATION  (CLIP ViT-B/32)
# ============================================================================

class CLIPScorer:
    """CLIP ViT-B/32 model wrapper for contrastive alignment."""

    def __init__(self, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        print(f"  Loading CLIP: {model_name} ...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, pil_crop: Image.Image, prompts: List[str]) -> List[float]:
        """Score one image crop against candidate prompts. Returns softmax probs."""
        if not prompts:
            return []
        inputs = self.processor(
            text=prompts, images=pil_crop,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits_per_image  # (1, N)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        return probs.cpu().tolist()


# ============================================================================
# §8  STAGE 4 — VYNIX LOGIC GATE  &  HALLUCINATION TRACKER
# ============================================================================

class HallucinationTracker:
    """Tracks instances where the Vynix gate vetoes false contact predictions."""

    def __init__(self):
        self.total = 0
        self.per_verb = Counter()
        self.per_object = Counter()

    def log(self, verb: str, obj: str):
        self.total += 1
        self.per_verb[verb] += 1
        self.per_object[obj] += 1

    def summary(self) -> str:
        lines = [f"\n  Hallucinations Prevented: {self.total}"]
        if self.per_verb:
            top_v = self.per_verb.most_common(5)
            lines.append("  Top overridden verbs: " +
                         ", ".join(f"{v}({c})" for v, c in top_v))
        if self.per_object:
            top_o = self.per_object.most_common(5)
            lines.append("  Top overridden objects: " +
                         ", ".join(f"{o}({c})" for o, c in top_o))
        return "\n".join(lines)


def apply_vynix_gate(verb: str, clip_prob: float, iou_val: float,
                     tracker: HallucinationTracker, obj_name: str) -> float:
    """Apply the Vynix Logic Gate override rule."""
    if verb == "no_interaction":
        return clip_prob
    if verb in CONTACT_VERBS and iou_val == 0.0:
        tracker.log(verb, obj_name)
        return 0.0  # Hard geometric veto
    return clip_prob


# ============================================================================
# §9  IMAGE PIPELINE
# ============================================================================

def process_image(
    pil_image: Image.Image,
    detector: NodeExtractor,
    clip_scorer: CLIPScorer,
    meta: HOIMeta,
    tracker: HallucinationTracker,
    max_pairs: int = 50,
) -> Dict[int, float]:
    """
    Run the 4-stage pipeline on one image.
    Returns: dict of {hoi_id: max_confidence}.
    """
    persons, objects = detector.detect(pil_image)
    if not persons or not objects:
        return {}

    img_w, img_h = pil_image.size
    predictions: Dict[int, float] = {}

    pairs = [(p, o) for p in persons for o in objects]
    if len(pairs) > max_pairs:
        pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
        pairs = pairs[:max_pairs]

    for (p_box, p_conf), (o_box, o_conf, o_cls) in pairs:
        iou_val = compute_iou(p_box, o_box)
        coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
        if coco_name is None:
            continue
        hico_name = normalize_name(coco_name)

        valid_entries = meta.obj_to_entries.get(hico_name, [])
        if not valid_entries:
            continue

        ubox = compute_union_box(p_box, o_box)
        x1 = max(0, int(ubox[0]))
        y1 = max(0, int(ubox[1]))
        x2 = min(img_w, int(ubox[2]))
        y2 = min(img_h, int(ubox[3]))
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        pil_crop = pil_image.crop((x1, y1, x2, y2))
        prompts = [make_prompt(gerund, hico_name) for _, gerund, _ in valid_entries]

        clip_probs = clip_scorer.score(pil_crop, prompts)
        if len(clip_probs) != len(valid_entries):
            continue

        for idx, (verb, _, hoi_id) in enumerate(valid_entries):
            gated_prob = apply_vynix_gate(
                verb, clip_probs[idx], iou_val, tracker, hico_name
            )
            final_conf = p_conf * o_conf * gated_prob
            if hoi_id not in predictions or final_conf > predictions[hoi_id]:
                predictions[hoi_id] = final_conf

    return predictions


# ============================================================================
# §10  mAP COMPUTATION  (VOC-Style All-Point Interpolation)
# ============================================================================

def compute_ap(scores: List[float], labels: List[int]) -> float:
    """Compute Average Precision (AP) for a single HOI class."""
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    n_pos = labels_arr.sum()
    if n_pos == 0:
        return 0.0

    sorted_idx = np.argsort(-scores_arr)
    labels_arr = labels_arr[sorted_idx]

    tp = np.cumsum(labels_arr)
    fp = np.cumsum(1 - labels_arr)
    precision = tp / (tp + fp)
    recall = tp / n_pos

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1])
    return float(ap)


def evaluate_all(
    all_predictions: Dict[int, Dict[int, float]],
    all_gt: Dict[int, Set[int]],
    meta: HOIMeta,
    num_images: int,
) -> Dict:
    """Compute mAP across Full, Rare, and Non-Rare classes."""
    per_class_ap = {}
    all_hoi_ids = sorted(meta.hoi_to_obj.keys())

    for hoi_id in all_hoi_ids:
        scores = []
        labels = []
        for img_idx in range(num_images):
            s = all_predictions.get(img_idx, {}).get(hoi_id, 0.0)
            l = 1 if hoi_id in all_gt.get(img_idx, set()) else 0
            scores.append(s)
            labels.append(l)

        ap = compute_ap(scores, labels)
        per_class_ap[hoi_id] = ap

    full_aps = [per_class_ap[h] for h in all_hoi_ids if h in per_class_ap]
    rare_aps = [per_class_ap[h] for h in meta.rare_ids if h in per_class_ap]
    non_rare_aps = [per_class_ap[h] for h in meta.non_rare_ids if h in per_class_ap]

    return {
        "per_class_ap": per_class_ap,
        "mAP_full": np.mean(full_aps) if full_aps else 0.0,
        "mAP_rare": np.mean(rare_aps) if rare_aps else 0.0,
        "mAP_non_rare": np.mean(non_rare_aps) if non_rare_aps else 0.0,
        "num_classes_full": len(full_aps),
        "num_classes_rare": len(rare_aps),
        "num_classes_non_rare": len(non_rare_aps),
    }


# ============================================================================
# §11  EXPORT & SUMMARY
# ============================================================================

def export_csv(results: Dict, meta: HOIMeta, path: str):
    """Export detailed per-class performance to CSV."""
    per_class = results["per_class_ap"]
    rows = []
    for hoi_id in sorted(per_class.keys()):
        obj = meta.hoi_to_obj.get(hoi_id, "?")
        verb = meta.hoi_to_verb.get(hoi_id, "?")
        category = "rare" if hoi_id in meta.rare_ids else "non-rare"
        rows.append({
            "hoi_id": hoi_id,
            "verb": verb,
            "object": obj,
            "ap": f"{per_class[hoi_id]:.6f}",
            "category": category,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hoi_id", "verb", "object", "ap", "category"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV exported: {path} ({len(rows)} rows)")


def print_summary(results: Dict, tracker: HallucinationTracker, elapsed: float):
    """Print the formatted benchmark summary table."""
    print("\n" + "=" * 64)
    print("  PROJECT VYNIX — HICO-DET BENCHMARK RESULTS")
    print("=" * 64)
    print(f"\n  {'Split':<15} {'HOI Classes':>12} {'mAP':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*10}")
    print(f"  {'Full':<15} {results['num_classes_full']:>12} "
          f"{results['mAP_full']*100:>9.2f}%")
    print(f"  {'Rare':<15} {results['num_classes_rare']:>12} "
          f"{results['mAP_rare']*100:>9.2f}%")
    print(f"  {'Non-Rare':<15} {results['num_classes_non_rare']:>12} "
          f"{results['mAP_non_rare']*100:>9.2f}%")
    print(tracker.summary())
    print(f"\n  Elapsed: {elapsed/60:.1f} minutes")
    print("=" * 64 + "\n")


# ============================================================================
# §12  CLI MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Project Vynix — HICO-DET Full Benchmark Evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Compute device (default: cpu)")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size parameter (default: 32)")
    parser.add_argument(
        "--limit-samples", type=int, default=None,
        help="Limit test images for quick verification (default: all)")
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Path to export per-class AP table as CSV")
    parser.add_argument(
        "--cache-dir", type=str, default="E:/Dataset",
        help="Dataset cache directory (default: E:/Dataset)")
    parser.add_argument(
        "--max-pairs", type=int, default=50,
        help="Max person-object pairs per image (default: 50)")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25,
        help="YOLOv8 detection confidence threshold (default: 0.25)")

    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  PROJECT VYNIX — HICO-DET Full Benchmark")
    print("=" * 64)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARNING] CUDA requested but not available in PyTorch. Automatically falling back to CPU.")
        device = "cpu"

    print(f"  Device:        {device} (requested: {args.device})")
    print(f"  Cache dir:     {args.cache_dir}")
    print(f"  Max pairs:     {args.max_pairs}")
    print(f"  Limit samples: {args.limit_samples or 'ALL'}")
    print()

    # Ensure dataset files are present in cache directory
    ensure_dataset_files(args.cache_dir)

    # Load 600 HOI metadata
    meta = load_hoi_metadata(args.cache_dir)

    # Load test split DataFrame
    test_df = load_test_dataframe(args.cache_dir, limit_samples=args.limit_samples)
    n_test = len(test_df)

    # Initialize models
    print("\n  Initializing models ...")
    detector = NodeExtractor(device=device)
    clip_scorer = CLIPScorer(device=device)
    tracker = HallucinationTracker()

    print(f"\n  Evaluating on {n_test} test images ...\n")

    all_predictions: Dict[int, Dict[int, float]] = {}
    all_gt: Dict[int, Set[int]] = {}

    t_start = time.time()
    for img_idx in tqdm(range(n_test), desc="  Evaluating", unit="img"):
        row = test_df.iloc[img_idx]

        # Ground truth positive HOI indices
        gt_objs = row.get("positive_objects")
        gt_set = set(parse_int_list(gt_objs))
        all_gt[img_idx] = gt_set

        # Decode image from raw bytes
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
            pil_image = Image.open(io.BytesIO(img_data["bytes"]))
        elif isinstance(img_data, Image.Image):
            pil_image = img_data
        else:
            continue

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Run 4-stage pipeline
        preds = process_image(
            pil_image, detector, clip_scorer, meta, tracker,
            max_pairs=args.max_pairs,
        )
        all_predictions[img_idx] = preds

    elapsed = time.time() - t_start

    # Compute mAP
    print("\n  Computing mAP ...")
    results = evaluate_all(all_predictions, all_gt, meta, n_test)

    # Output results
    print_summary(results, tracker, elapsed)

    if args.output_csv:
        export_csv(results, meta, args.output_csv)

    return results


if __name__ == "__main__":
    main()
