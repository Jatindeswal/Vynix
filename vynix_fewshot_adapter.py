#!/usr/bin/env python3
"""
Project Vynix — Few-Shot CLIP Adapter (Tip-Adapter-F)
=====================================================
Builds a lightweight learnable cache on top of frozen CLIP ViT-B/32 to
push the zero-shot 22.17% mAP baseline higher using K-shot examples.

Architecture (Tip-Adapter-F):
    f_visual    = CLIP_vision(union_crop)                    # (1, 512) frozen
    affinity    = exp(-beta * (1 - f_visual @ cache_keys.T)) # (1, N)
    cache_logit = affinity @ cache_values                    # (1, 600)
    clip_logit  = f_visual @ text_weights.T                  # (1, 600)
    final       = clip_logit + alpha * cache_logit            # residual blend

Only alpha, beta (and optionally cache_keys) are trained.  CLIP stays frozen.

Usage:
    python vynix_fewshot_adapter.py --dataset-dir E:\\Dataset --k-shots 1 5 10 \\
        --device cuda --eval-limit 500 --output-dir fewshot_results
"""

import argparse
import csv
import glob
import io
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    sys.exit("Install: pip install ultralytics transformers")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # charts optional


# ═══════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

CONTACT_VERBS = {
    "hold","carry","hug","kiss","lick","eat","drink_with","sip","taste",
    "wear","ride","sit_on","sit_at","lie_on","stand_on","straddle","pet",
    "groom","milk","shear","touch","catch","grab","pick_up","pick","lift",
    "flip","push","pull","cut","cut_with","hit","kick","tie","wash","dry",
    "brush_with","fill","pour","stab","squeeze","type_on","wield","swing",
    "operate","play_with","control","drive","fly","row","sail","board",
    "hop_on","mount","drag","dribble","grind","hose","load","open","pack",
    "peel","spin","zip",
}


# ═══════════════════════════════════════════════════════════════════════════
# §2  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    name = str(name).lower().strip().replace("_", " ")
    aliases = {
        "hair dryer":"hair drier","tv monitor":"tv","television":"tv",
        "motorbike":"motorcycle","sofa":"couch","aeroplane":"airplane",
        "cell_phone":"cell phone","hot_dog":"hot dog",
        "wine_glass":"wine glass","potted_plant":"potted plant",
        "dining_table":"dining table","teddy_bear":"teddy bear",
        "sports_ball":"sports ball","baseball_bat":"baseball bat",
        "baseball_glove":"baseball glove","tennis_racket":"tennis racket",
        "fire_hydrant":"fire hydrant","stop_sign":"stop sign",
        "parking_meter":"parking meter","traffic_light":"traffic light",
    }
    return aliases.get(name, name)

def parse_int_list(val) -> List[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [int(x) for x in val]
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "[]":
            return []
        return [int(x) for x in re.findall(r"\d+", val)]
    return []

def make_prompt(gerund: str, obj: str) -> str:
    g = gerund.replace("_", " ").strip()
    o = obj.replace("_", " ").strip()
    art = "an" if o and o[0].lower() in "aeiou" else "a"
    if g == "no interaction":
        return f"a person standing near {art} {o} without interacting"
    return f"a person {g} {art} {o}"

def compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0

def union_box(a, b):
    return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]


# ═══════════════════════════════════════════════════════════════════════════
# §3  HOI METADATA
# ═══════════════════════════════════════════════════════════════════════════

class HOIMeta:
    def __init__(self, dataset_dir: str):
        csv_path = os.path.join(dataset_dir, "list_action.csv")
        df = pd.read_csv(csv_path)
        self.num_classes = len(df)
        self.hoi_to_obj: Dict[int, str] = {}
        self.hoi_to_verb: Dict[int, str] = {}
        self.hoi_to_gerund: Dict[int, str] = {}
        self.obj_to_entries: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)

        for idx, row in df.iterrows():
            hoi_id = int(idx)
            obj = normalize_name(str(row["nname"]))
            verb = str(row["vname"]).strip()
            ger = str(row["vname_ing"]).strip() if pd.notna(row.get("vname_ing")) else verb + "ing"
            self.hoi_to_obj[hoi_id] = obj
            self.hoi_to_verb[hoi_id] = verb
            self.hoi_to_gerund[hoi_id] = ger
            self.obj_to_entries[obj].append((verb, ger, hoi_id))


# ═══════════════════════════════════════════════════════════════════════════
# §4  FEATURE EXTRACTOR  (Frozen CLIP + YOLO)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extracts 512-d CLIP visual features from union crops."""

    def __init__(self, device: str):
        self.device = device
        print("  Loading YOLOv8-nano...")
        self.yolo = YOLO("yolov8n.pt")
        print("  Loading CLIP ViT-B/32...")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    @staticmethod
    def _safe_extract(feats):
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            return feats.pooler_output
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            return feats.text_embeds
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            return feats.image_embeds
        if isinstance(feats, (tuple, list)):
            return feats[0]
        return feats

    @torch.no_grad()
    def extract_visual_feature(self, pil_crop: Image.Image) -> torch.Tensor:
        """Returns (512,) normalized visual embedding."""
        inputs = self.processor(images=pil_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self._safe_extract(self.model.get_image_features(**inputs))  # (1, 512)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()                                       # (512,)

    def detect(self, pil_image: Image.Image):
        """Returns (persons, objects) lists."""
        results = self.yolo(pil_image, device=self.device, verbose=False)
        persons, objects = [], []
        for r in results:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                if conf < 0.25:
                    continue
                box = r.boxes.xyxy[i].tolist()
                if cls_id == 0:
                    persons.append((box, conf))
                else:
                    objects.append((box, conf, cls_id))
        return persons, objects

    @torch.no_grad()
    def build_text_weights(self, meta: HOIMeta) -> torch.Tensor:
        """Build (num_classes, 512) text classifier weight matrix."""
        prompts = []
        for hoi_id in range(meta.num_classes):
            ger = meta.hoi_to_gerund[hoi_id]
            obj = meta.hoi_to_obj[hoi_id]
            prompts.append(make_prompt(ger, obj))

        # Process in batches to avoid OOM
        all_feats = []
        bs = 64
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i+bs]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._safe_extract(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)  # (600, 512)


# ═══════════════════════════════════════════════════════════════════════════
# §5  TIP-ADAPTER-F  (Learnable Few-Shot Cache)
# ═══════════════════════════════════════════════════════════════════════════

class TipAdapterF(nn.Module):
    """
    Tip-Adapter-F: Training-Free CLIP-Adapter with learnable parameters.

    Math:
        affinity   = exp(-beta * (1 - f @ cache_keys.T))
        cache_logit = affinity @ cache_values
        final      = clip_logit + alpha * cache_logit
    """

    def __init__(self, cache_keys: torch.Tensor, cache_values: torch.Tensor):
        super().__init__()
        # cache_keys: (N, 512), cache_values: (N, C)
        self.cache_keys = nn.Parameter(cache_keys.clone())
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(5.5))
        self.register_buffer("cache_values", cache_values)

    def forward(self, clip_logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_logits: (B, C) from text classifier
            features:    (B, 512) visual features
        Returns:
            (B, C) blended logits
        """
        # Affinity between test features and cached training features
        affinity = features @ self.cache_keys.T                       # (B, N)
        affinity = (-self.beta * (1.0 - affinity)).exp()              # temperature
        cache_logits = affinity @ self.cache_values                   # (B, C)
        return clip_logits + self.alpha * cache_logits


# ═══════════════════════════════════════════════════════════════════════════
# §6  CACHE BUILDER  (K-Shot Sampling from Training Set)
# ═══════════════════════════════════════════════════════════════════════════

def build_few_shot_cache(
    train_files: List[str],
    meta: HOIMeta,
    extractor: FeatureExtractor,
    k_shot: int,
    max_images: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample K positive training images per HOI class and extract CLIP features.

    Returns:
        cache_keys:   (N, 512)  visual feature vectors
        cache_values: (N, C)    one-hot label vectors
    """
    print(f"\n  Building {k_shot}-shot cache...")

    # Step 1: Index which images are positive for each HOI class
    class_to_images: Dict[int, List[int]] = defaultdict(list)
    all_rows = []

    for f in train_files:
        df = pd.read_parquet(f)
        for i, row in df.iterrows():
            pos = parse_int_list(row.get("positive_objects"))
            for hoi_id in pos:
                if hoi_id < meta.num_classes:
                    class_to_images[hoi_id].append(len(all_rows))
            all_rows.append(row)
            if len(all_rows) >= max_images:
                break
        if len(all_rows) >= max_images:
            break

    print(f"    Indexed {len(all_rows)} training images, {len(class_to_images)} classes with positives.")

    # Step 2: Sample K images per class
    sampled_indices = set()
    class_samples: Dict[int, List[int]] = {}
    rng = np.random.RandomState(42)

    for hoi_id in range(meta.num_classes):
        candidates = class_to_images.get(hoi_id, [])
        if not candidates:
            continue
        k = min(k_shot, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False).tolist()
        class_samples[hoi_id] = chosen
        sampled_indices.update(chosen)

    print(f"    Selected {len(sampled_indices)} unique images for cache extraction.")

    # Step 3: Extract CLIP features for each sampled image's union crop
    cache_keys_list = []
    cache_values_list = []
    processed = 0

    for img_idx in tqdm(sorted(sampled_indices), desc="    Extracting cache features", unit="img"):
        row = all_rows[img_idx]
        pos_set = set(parse_int_list(row.get("positive_objects")))

        # Decode image
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
            pil_img = Image.open(io.BytesIO(img_data["bytes"]))
        elif isinstance(img_data, Image.Image):
            pil_img = img_data
        else:
            continue
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Detect persons and objects
        persons, objects = extractor.detect(pil_img)
        if not persons or not objects:
            continue

        img_w, img_h = pil_img.size

        # For each valid person-object pair, extract feature
        for (p_box, _), (o_box, _, o_cls) in [(p, o) for p in persons[:3] for o in objects[:5]]:
            coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
            if not coco_name:
                continue
            hico_name = normalize_name(coco_name)
            entries = meta.obj_to_entries.get(hico_name, [])
            if not entries:
                continue

            # Check if any HOI class for this object is actually positive
            relevant_hoi_ids = [hid for _, _, hid in entries if hid in pos_set]
            if not relevant_hoi_ids:
                continue

            ubox = union_box(p_box, o_box)
            x1 = max(0, int(ubox[0]))
            y1 = max(0, int(ubox[1]))
            x2 = min(img_w, int(ubox[2]))
            y2 = min(img_h, int(ubox[3]))
            if (x2-x1) < 10 or (y2-y1) < 10:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            feat = extractor.extract_visual_feature(crop)  # (512,)

            # Build one-hot label
            label = torch.zeros(meta.num_classes)
            for hid in relevant_hoi_ids:
                label[hid] = 1.0

            cache_keys_list.append(feat)
            cache_values_list.append(label)
            processed += 1
            break  # One crop per image is enough

    if not cache_keys_list:
        print("    ⚠ No cache entries found! Returning empty cache.")
        return torch.zeros(1, 512), torch.zeros(1, meta.num_classes)

    keys = torch.stack(cache_keys_list)    # (N, 512)
    values = torch.stack(cache_values_list)  # (N, C)
    print(f"    ✓ Cache built: {keys.shape[0]} entries × {keys.shape[1]}-d features")
    return keys, values


# ═══════════════════════════════════════════════════════════════════════════
# §7  ADAPTER TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_adapter(
    adapter: TipAdapterF,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    text_weights: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
) -> List[float]:
    """
    Fine-tune alpha, beta (and cache_keys) for a few epochs.
    Returns list of per-epoch losses.
    """
    adapter = adapter.to(device)
    text_weights = text_weights.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for epoch in range(epochs):
        adapter.train()
        # Forward pass (full batch — cache is small enough)
        clip_logits = train_features @ text_weights.T     # (N, C)
        final_logits = adapter(clip_logits, train_features)  # (N, C)

        loss = F.binary_cross_entropy_with_logits(final_logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"alpha={adapter.alpha.item():.3f}, beta={adapter.beta.item():.3f}")

    adapter.eval()
    return losses


# ═══════════════════════════════════════════════════════════════════════════
# §8  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def compute_ap(scores: List[float], labels: List[int]) -> float:
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)
    n_pos = labels_arr.sum()
    if n_pos == 0:
        return 0.0
    idx = np.argsort(-scores_arr)
    labels_arr = labels_arr[idx]
    tp = np.cumsum(labels_arr)
    fp = np.cumsum(1 - labels_arr)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    ch = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ch+1] - mrec[ch]) * mpre[ch+1]))


def evaluate_adapter(
    adapter: TipAdapterF,
    extractor: FeatureExtractor,
    text_weights: torch.Tensor,
    meta: HOIMeta,
    test_files: List[str],
    device: str,
    eval_limit: int = None,
) -> Dict:
    """Run full HICO-DET evaluation with the adapter."""
    adapter = adapter.to(device).eval()
    text_weights_dev = text_weights.to(device)

    # Load test data
    dfs = [pd.read_parquet(f) for f in test_files]
    test_df = pd.concat(dfs, ignore_index=True)
    if eval_limit:
        test_df = test_df.iloc[:eval_limit]

    n_images = len(test_df)
    all_preds: Dict[int, Dict[int, float]] = {}
    all_gt: Dict[int, Set[int]] = {}
    total_vetoes = 0

    for img_idx in tqdm(range(n_images), desc="    Evaluating", unit="img"):
        row = test_df.iloc[img_idx]
        all_gt[img_idx] = set(parse_int_list(row.get("positive_objects")))

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
        preds: Dict[int, float] = {}

        pairs = [(p, o) for p in persons for o in objects]
        if len(pairs) > 50:
            pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
            pairs = pairs[:50]

        for (p_box, p_conf), (o_box, o_conf, o_cls) in pairs:
            iou_val = compute_iou(p_box, o_box)
            coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
            if not coco_name:
                continue
            hico_name = normalize_name(coco_name)
            entries = meta.obj_to_entries.get(hico_name, [])
            if not entries:
                continue

            ubox = union_box(p_box, o_box)
            x1 = max(0, int(ubox[0]))
            y1 = max(0, int(ubox[1]))
            x2 = min(img_w, int(ubox[2]))
            y2 = min(img_h, int(ubox[3]))
            if (x2-x1) < 10 or (y2-y1) < 10:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            feat = extractor.extract_visual_feature(crop).unsqueeze(0).to(device)  # (1, 512)

            with torch.no_grad():
                clip_logits = feat @ text_weights_dev.T  # (1, 600)
                final_logits = adapter(clip_logits, feat)  # (1, 600)
                probs = torch.softmax(final_logits, dim=1).squeeze(0).cpu()

            for verb, _, hoi_id in entries:
                raw_prob = probs[hoi_id].item()

                # Vynix Logic Gate: geometric override
                if verb in CONTACT_VERBS and iou_val == 0.0:
                    gated_prob = 0.0
                    total_vetoes += 1
                else:
                    gated_prob = raw_prob

                conf = p_conf * o_conf * gated_prob
                if hoi_id not in preds or conf > preds[hoi_id]:
                    preds[hoi_id] = conf

        all_preds[img_idx] = preds

    # Compute mAP
    all_hoi_ids = sorted(meta.hoi_to_obj.keys())
    per_class_ap = {}
    for hoi_id in all_hoi_ids:
        scores = [all_preds.get(i, {}).get(hoi_id, 0.0) for i in range(n_images)]
        labels = [1 if hoi_id in all_gt.get(i, set()) else 0 for i in range(n_images)]
        per_class_ap[hoi_id] = compute_ap(scores, labels)

    full_aps = [per_class_ap[h] for h in all_hoi_ids]
    # Approximate rare/non-rare split
    rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}
    nonrare_ids = set(range(meta.num_classes)) - rare_ids
    rare_aps = [per_class_ap[h] for h in rare_ids if h in per_class_ap]
    nonrare_aps = [per_class_ap[h] for h in nonrare_ids if h in per_class_ap]

    return {
        "mAP_full": float(np.mean(full_aps)) * 100,
        "mAP_rare": float(np.mean(rare_aps)) * 100 if rare_aps else 0.0,
        "mAP_non_rare": float(np.mean(nonrare_aps)) * 100 if nonrare_aps else 0.0,
        "vetoes": total_vetoes,
        "n_images": n_images,
        "per_class_ap": per_class_ap,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §9  CHART GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_charts(results: Dict, all_losses: Dict, output_dir: str):
    if plt is None:
        print("  ⚠ matplotlib not available, skipping charts.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: mAP comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    k_values = sorted(results.keys())
    splits = ["Full (600)", "Rare", "Non-Rare"]
    x = np.arange(len(splits))
    width = 0.8 / len(k_values)
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, k in enumerate(k_values):
        vals = [results[k]["mAP_full"], results[k]["mAP_rare"], results[k]["mAP_non_rare"]]
        bars = ax.bar(x + i * width - (len(k_values)-1)*width/2, vals, width,
                      label=f"{k}-shot", color=colors[i % len(colors)], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("mAP (%)", fontsize=12, fontweight="bold")
    ax.set_title("Few-Shot Adapter Performance: Zero-Shot vs K-Shot", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fewshot_comparison_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved fewshot_comparison_bar.png")

    # Chart 2: Training loss curves
    if all_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, losses in sorted(all_losses.items()):
            if losses:
                ax.plot(range(1, len(losses)+1), losses, marker="o", markersize=4, label=f"{k}-shot")
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("BCE Loss", fontsize=12, fontweight="bold")
        ax.set_title("Tip-Adapter-F Training Loss Curves", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fewshot_training_loss.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved fewshot_training_loss.png")


# ═══════════════════════════════════════════════════════════════════════════
# §10  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vynix Few-Shot Adapter (Tip-Adapter-F)")
    parser.add_argument("--dataset-dir", type=str, default=r"E:\Dataset",
                        help="Path to HICO-DET dataset directory")
    parser.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 10],
                        help="K-shot values to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-limit", type=int, default=500,
                        help="Max test images (None = full 9658)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="fewshot_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  PROJECT VYNIX — FEW-SHOT CLIP ADAPTER (Tip-Adapter-F)")
    print("=" * 70)

    # Load metadata
    meta = HOIMeta(args.dataset_dir)
    print(f"  ✓ Loaded {meta.num_classes} HOI classes.")

    # Initialize extractor
    extractor = FeatureExtractor(args.device)

    # Build text classifier weights
    print("\n  Building text classifier weights...")
    text_weights = extractor.build_text_weights(meta)  # (600, 512)
    print(f"  ✓ Text weights: {text_weights.shape}")

    # File paths
    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))
    print(f"  ✓ Train shards: {len(train_files)}, Test shards: {len(test_files)}")

    # Zero-shot baseline (k=0, no adapter)
    all_results = {}
    all_losses = {}

    print("\n" + "─" * 70)
    print("  ▶ Evaluating 0-shot (Zero-Shot Baseline, no adapter)...")
    print("─" * 70)

    # For zero-shot, create a dummy adapter with empty cache
    dummy_keys = torch.zeros(1, 512)
    dummy_vals = torch.zeros(1, meta.num_classes)
    dummy_adapter = TipAdapterF(dummy_keys, dummy_vals)
    dummy_adapter.alpha.data.fill_(0.0)  # alpha=0 means purely text-based

    result_0 = evaluate_adapter(dummy_adapter, extractor, text_weights, meta,
                                test_files, args.device, args.eval_limit)
    all_results[0] = result_0
    print(f"\n  0-shot: Full={result_0['mAP_full']:.2f}%  Rare={result_0['mAP_rare']:.2f}%  "
          f"Non-Rare={result_0['mAP_non_rare']:.2f}%  Vetoes={result_0['vetoes']}")

    # K-shot runs
    for k in args.k_shots:
        print(f"\n{'─' * 70}")
        print(f"  ▶ Building and evaluating {k}-shot adapter...")
        print("─" * 70)

        # Build cache
        cache_keys, cache_values = build_few_shot_cache(
            train_files, meta, extractor, k, max_images=5000
        )

        # Create adapter
        adapter = TipAdapterF(cache_keys, cache_values)

        # Train
        print(f"\n    Training Tip-Adapter-F ({k}-shot, {args.epochs} epochs)...")
        losses = train_adapter(adapter, cache_keys, cache_values, text_weights,
                               epochs=args.epochs, device=args.device)
        all_losses[k] = losses

        # Evaluate
        result_k = evaluate_adapter(adapter.cpu(), extractor, text_weights, meta,
                                    test_files, args.device, args.eval_limit)
        all_results[k] = result_k
        print(f"\n  {k}-shot: Full={result_k['mAP_full']:.2f}%  Rare={result_k['mAP_rare']:.2f}%  "
              f"Non-Rare={result_k['mAP_non_rare']:.2f}%  Vetoes={result_k['vetoes']}")

    # Final results table
    print("\n\n" + "=" * 70)
    print("  🏆 VYNIX FEW-SHOT ADAPTER RESULTS (Tip-Adapter-F)")
    print("=" * 70)
    print(f"  {'K-Shot':<8} | {'Full mAP':>10} | {'Rare mAP':>10} | {'Non-Rare':>10} | {'Delta':>10}")
    print("─" * 70)

    baseline_full = all_results[0]["mAP_full"]
    for k in sorted(all_results.keys()):
        r = all_results[k]
        delta = r["mAP_full"] - baseline_full
        sign = "+" if delta >= 0 else ""
        tag = "baseline" if k == 0 else f"{sign}{delta:.2f}%"
        print(f"  {k:<8} | {r['mAP_full']:>9.2f}% | {r['mAP_rare']:>9.2f}% | "
              f"{r['mAP_non_rare']:>9.2f}% | {tag:>10}")

    print("=" * 70)

    # Generate charts
    print("\n  Generating charts...")
    generate_charts(all_results, all_losses, args.output_dir)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "fewshot_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k_shot", "mAP_full", "mAP_rare", "mAP_non_rare", "vetoes", "n_images"])
        writer.writeheader()
        for k in sorted(all_results.keys()):
            r = all_results[k]
            writer.writerow({
                "k_shot": k,
                "mAP_full": f"{r['mAP_full']:.4f}",
                "mAP_rare": f"{r['mAP_rare']:.4f}",
                "mAP_non_rare": f"{r['mAP_non_rare']:.4f}",
                "vetoes": r["vetoes"],
                "n_images": r["n_images"],
            })
    print(f"  ✓ Results saved to {csv_path}")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
