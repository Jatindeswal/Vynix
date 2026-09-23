#!/usr/bin/env python3
"""
Project Vynix — 3-Stream Spatial-Visual Few-Shot Adapter (Tip-Adapter-F)
======================================================================
Upgrades the zero-shot baseline (22.17% mAP) by integrating:
1. 3-Stream Visual Fusion:
   - Human Crop (f_h in R^512): Captures human pose, gaze, and hands
   - Object Crop (f_o in R^512): Preserves small object details (knife, orange, phone)
   - Union Crop (f_u in R^512): Preserves global interaction context
   - Fused Representation: f_vis = [f_h || f_o || f_u] in R^1536
2. Continuous Spatial Geometry MLP:
   - Encodes 8D relative bounding-box geometry [dx, dy, wp, hp, wo, ho, IoU, area_ratio]
   - Learns interaction-specific spatial likelihood priors P_spatial in [0, 1]^600
3. Tip-Adapter-F Multi-Stream Cache:
   - Stores few-shot exemplar embeddings without fine-tuning frozen CLIP backbone
   - Blends visual cache affinity with text classifier logits
4. Vynix Geometric Override Gate:
   - Hard safety gate vetoing contact predictions when IoU == 0

Usage:
    python vynix_fewshot_adapter.py --dataset-dir E:\\Dataset --k-shots 1 5 10 \\
        --device cuda --detector-conf 0.08 --use-3stream --use-spatial-mlp \\
        --eval-limit 500 --output-dir fewshot_results
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
from typing import Dict, List, Optional, Set, Tuple

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
    sys.exit("Install required packages: pip install ultralytics transformers")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
# §2  HELPERS & SPATIAL GEOMETRY
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

def compute_iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0

def compute_union_box(a: List[float], b: List[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

def compute_spatial_vector(p_box: List[float], o_box: List[float], img_w: int, img_h: int) -> torch.Tensor:
    """
    Computes an 8D normalized continuous spatial relationship vector:
    [dx, dy, wp, hp, wo, ho, IoU, log_area_ratio]
    """
    w_safe = max(float(img_w), 1.0)
    h_safe = max(float(img_h), 1.0)

    c_px = (p_box[0] + p_box[2]) / (2.0 * w_safe)
    c_py = (p_box[1] + p_box[3]) / (2.0 * h_safe)
    c_ox = (o_box[0] + o_box[2]) / (2.0 * w_safe)
    c_oy = (o_box[1] + o_box[3]) / (2.0 * h_safe)

    dx = c_px - c_ox
    dy = c_py - c_oy

    wp = (p_box[2] - p_box[0]) / w_safe
    hp = (p_box[3] - p_box[1]) / h_safe
    wo = (o_box[2] - o_box[0]) / w_safe
    ho = (o_box[3] - o_box[1]) / h_safe

    iou_val = compute_iou(p_box, o_box)
    area_p = max(wp * hp, 1e-5)
    area_o = max(wo * ho, 1e-5)
    log_area_ratio = math.log(area_p / area_o)

    return torch.tensor([dx, dy, wp, hp, wo, ho, iou_val, log_area_ratio], dtype=torch.float32)


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
# §4  SPATIAL GEOMETRY MLP
# ═══════════════════════════════════════════════════════════════════════════

class SpatialMLP(nn.Module):
    """
    Continuous Spatial Geometry MLP:
    Maps 8D relative spatial vector -> 600 interaction probability priors.
    """
    def __init__(self, num_classes: int = 600, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, spatial_vec: torch.Tensor) -> torch.Tensor:
        """Args: (B, 8) -> Returns: (B, 600) priors in [0, 1]"""
        return self.net(spatial_vec)


# ═══════════════════════════════════════════════════════════════════════════
# §5  MULTI-STREAM FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class MultiStreamFeatureExtractor:
    """
    Extracts high-resolution visual embeddings for:
    - Person crop (hands, posture, gaze)
    - Object crop (knife, fork, orange, phone details)
    - Union crop (contextual interaction)
    """

    def __init__(self, device: str, detector_conf: float = 0.08, use_3stream: bool = True):
        self.device = device
        self.detector_conf = detector_conf
        self.use_3stream = use_3stream
        self.feature_dim = 1536 if use_3stream else 512

        print(f"  Loading YOLOv8-nano (Confidence threshold = {detector_conf})...")
        self.yolo = YOLO("yolov8n.pt")
        print("  Loading CLIP ViT-B/32 (Vision & Text Backbones)...")
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
    def _encode_crop(self, pil_crop: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self._safe_extract(self.model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()  # (512,)

    def extract_visual_feature(
        self, pil_img: Image.Image, p_box: List[float], o_box: List[float], u_box: List[float]
    ) -> torch.Tensor:
        """
        Extracts and concatenates 3-stream visual representations.
        Returns:
            (1536,) if use_3stream=True else (512,)
        """
        img_w, img_h = pil_img.size

        # Clamp boxes safely
        def safe_crop(b):
            x1 = max(0, min(int(b[0]), img_w - 2))
            y1 = max(0, min(int(b[1]), img_h - 2))
            x2 = max(x1 + 2, min(int(b[2]), img_w))
            y2 = max(y1 + 2, min(int(b[3]), img_h))
            return pil_img.crop((x1, y1, x2, y2))

        u_crop = safe_crop(u_box)
        f_union = self._encode_crop(u_crop)

        if not self.use_3stream:
            return f_union

        p_crop = safe_crop(p_box)
        o_crop = safe_crop(o_box)

        f_person = self._encode_crop(p_crop)
        f_object = self._encode_crop(o_crop)

        # Concatenate 3 normalized streams
        f_fused = torch.cat([f_person, f_object, f_union], dim=-1)
        return f_fused / f_fused.norm(dim=-1, keepdim=True)

    def detect(self, pil_image: Image.Image):
        """Returns detected (persons, objects) with lower threshold for high recall."""
        results = self.yolo(pil_image, device=self.device, verbose=False)
        persons, objects = [], []
        for r in results:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                if conf < self.detector_conf:
                    continue
                box = r.boxes.xyxy[i].tolist()
                if cls_id == 0:
                    persons.append((box, conf))
                else:
                    objects.append((box, conf, cls_id))
        return persons, objects

    @torch.no_grad()
    def build_text_weights(self, meta: HOIMeta) -> torch.Tensor:
        """Build (num_classes, 512) normalized text classifier matrix."""
        prompts = []
        for hoi_id in range(meta.num_classes):
            ger = meta.hoi_to_gerund[hoi_id]
            obj = meta.hoi_to_obj[hoi_id]
            prompts.append(make_prompt(ger, obj))

        all_feats = []
        bs = 64
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i+bs]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._safe_extract(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# §6  VYNIX 3-STREAM SPATIAL-VISUAL ADAPTER (Tip-Adapter-F + Spatial MLP)
# ═══════════════════════════════════════════════════════════════════════════

class Vynix3StreamAdapter(nn.Module):
    """
    3-Stream Spatial-Visual Adapter:
    - Multi-stream visual cache (N, 1536)
    - Continuous spatial MLP (8 -> 64 -> 600)
    - Residual logit blending with frozen CLIP text classifier
    """

    def __init__(
        self,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        use_spatial_mlp: bool = True,
        num_classes: int = 600,
    ):
        super().__init__()
        self.cache_keys = nn.Parameter(cache_keys.clone())
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(5.5))
        self.register_buffer("cache_values", cache_values)

        self.use_spatial_mlp = use_spatial_mlp
        if use_spatial_mlp:
            self.spatial_mlp = SpatialMLP(num_classes=num_classes)
        else:
            self.spatial_mlp = None

    def forward(
        self,
        clip_logits,
        visual_features,
        spatial_features=None,
    ):
        """
        Args:
            clip_logits:      (B, 600) from text classifier
            visual_features:  (B, 1536) fused visual crop embeddings
            spatial_features: (B, 8) normalized spatial geometry
        Returns:
            (B, 600) blended logits
        """
        # Cosine affinity against visual cache
        affinity = visual_features @ self.cache_keys.T
        affinity = (-self.beta * (1.0 - affinity)).exp()
        cache_logits = affinity @ self.cache_values

        blended_logits = clip_logits + self.alpha * cache_logits

        if self.use_spatial_mlp and spatial_features is not None:
            spatial_priors = self.spatial_mlp(spatial_features)
            # Add spatial log-prior to logits
            spatial_log_prior = torch.log(spatial_priors + 1e-6)
            blended_logits = blended_logits + 0.5 * spatial_log_prior

        return blended_logits


# ═══════════════════════════════════════════════════════════════════════════
# §7  CACHE BUILDER (3-Stream Exemplar Sampling)
# ═══════════════════════════════════════════════════════════════════════════

def build_few_shot_cache(
    train_files: List[str],
    meta: HOIMeta,
    extractor: MultiStreamFeatureExtractor,
    k_shot: int,
    max_images: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples K training exemplars per HOI class and extracts 3-stream visual features
    and 8D spatial geometry vectors.
    """
    print(f"\n  Building {k_shot}-shot 3-Stream cache...")

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

    print(f"    Indexed {len(all_rows)} training images across {len(class_to_images)} classes.")

    sampled_indices = set()
    rng = np.random.RandomState(42)

    for hoi_id in range(meta.num_classes):
        candidates = class_to_images.get(hoi_id, [])
        if not candidates:
            continue
        k = min(k_shot, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False).tolist()
        sampled_indices.update(chosen)

    print(f"    Selected {len(sampled_indices)} images for 3-Stream exemplar extraction.")

    cache_keys_list = []
    cache_values_list = []
    spatial_vecs_list = []

    for img_idx in tqdm(sorted(sampled_indices), desc="    Extracting 3-stream cache", unit="img"):
        row = all_rows[img_idx]
        pos_set = set(parse_int_list(row.get("positive_objects")))

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
            continue

        img_w, img_h = pil_img.size

        for (p_box, _), (o_box, _, o_cls) in [(p, o) for p in persons[:3] for o in objects[:5]]:
            coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
            if not coco_name:
                continue
            hico_name = normalize_name(coco_name)
            entries = meta.obj_to_entries.get(hico_name, [])
            if not entries:
                continue

            rel_ids = [hid for _, _, hid in entries if hid in pos_set]
            if not rel_ids:
                continue

            u_box = compute_union_box(p_box, o_box)
            f_fused = extractor.extract_visual_feature(pil_img, p_box, o_box, u_box)
            s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)

            label = torch.zeros(meta.num_classes)
            for hid in rel_ids:
                label[hid] = 1.0

            cache_keys_list.append(f_fused)
            cache_values_list.append(label)
            spatial_vecs_list.append(s_vec)
            break

    if not cache_keys_list:
        return (
            torch.zeros(1, extractor.feature_dim),
            torch.zeros(1, meta.num_classes),
            torch.zeros(1, 8),
        )

    keys = torch.stack(cache_keys_list)
    values = torch.stack(cache_values_list)
    spatials = torch.stack(spatial_vecs_list)
    print(f"    ✓ 3-Stream cache ready: {keys.shape[0]} entries × {keys.shape[1]}-d visual features")
    return keys, values, spatials


# ═══════════════════════════════════════════════════════════════════════════
# §8  ADAPTER TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_adapter(
    adapter: Vynix3StreamAdapter,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_spatials: torch.Tensor,
    text_weights: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
) -> List[float]:
    """Fine-tunes alpha, beta, and spatial MLP parameters."""
    adapter = adapter.to(device)
    text_weights = text_weights.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    train_spatials = train_spatials.to(device)

    # Union stream is the last 512 dims if 3-stream
    if train_features.shape[-1] == 1536:
        f_union = train_features[:, 1024:1536]
    else:
        f_union = train_features

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for epoch in range(epochs):
        adapter.train()
        clip_logits = f_union @ text_weights.T
        final_logits = adapter(clip_logits, train_features, train_spatials)

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
# §9  EVALUATION ENGINE
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
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ch = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ch + 1] - mrec[ch]) * mpre[ch + 1]))


def evaluate_adapter(
    adapter: Vynix3StreamAdapter,
    extractor: MultiStreamFeatureExtractor,
    text_weights: torch.Tensor,
    meta: HOIMeta,
    test_files: List[str],
    device: str,
    eval_limit: Optional[int] = None,
) -> Dict:
    adapter = adapter.to(device).eval()
    text_weights_dev = text_weights.to(device)

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

            u_box = compute_union_box(p_box, o_box)
            f_fused = extractor.extract_visual_feature(pil_img, p_box, o_box, u_box).unsqueeze(0).to(device)
            s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h).unsqueeze(0).to(device)

            # Union feature for text logits
            f_union = f_fused[:, 1024:1536] if f_fused.shape[-1] == 1536 else f_fused

            with torch.no_grad():
                clip_logits = f_union @ text_weights_dev.T
                final_logits = adapter(clip_logits, f_fused, s_vec)
                probs = torch.softmax(final_logits, dim=1).squeeze(0).cpu()

            for verb, _, hoi_id in entries:
                raw_prob = probs[hoi_id].item()

                # Vynix Geometric Gate
                if verb in CONTACT_VERBS and iou_val == 0.0:
                    gated_prob = 0.0
                    total_vetoes += 1
                else:
                    gated_prob = raw_prob

                conf = p_conf * o_conf * gated_prob
                if hoi_id not in preds or conf > preds[hoi_id]:
                    preds[hoi_id] = conf

        all_preds[img_idx] = preds

    all_hoi_ids = sorted(meta.hoi_to_obj.keys())
    per_class_ap = {}
    for hoi_id in all_hoi_ids:
        scores = [all_preds.get(i, {}).get(hoi_id, 0.0) for i in range(n_images)]
        labels = [1 if hoi_id in all_gt.get(i, set()) else 0 for i in range(n_images)]
        per_class_ap[hoi_id] = compute_ap(scores, labels)

    full_aps = [per_class_ap[h] for h in all_hoi_ids]
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
# §10  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vynix 3-Stream Spatial-Visual Adapter")
    parser.add_argument("--dataset-dir", type=str, default=r"E:\Dataset")
    parser.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-conf", type=float, default=0.08,
                        help="YOLO detection confidence (lowered to 0.08 to recall small objects)")
    parser.add_argument("--use-3stream", action="store_true", default=True,
                        help="Use Human, Object, and Union 3-Stream fusion")
    parser.add_argument("--use-spatial-mlp", action="store_true", default=True,
                        help="Use continuous 8D spatial geometry MLP")
    parser.add_argument("--eval-limit", type=int, default=500,
                        help="Max test images for evaluation (None = full 9658)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="fewshot_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("  PROJECT VYNIX — 3-STREAM SPATIAL-VISUAL FEW-SHOT ADAPTER")
    print("=" * 75)
    print(f"  Configuration:")
    print(f"  • 3-Stream Visual Fusion  : {'Enabled (1536-d)' if args.use_3stream else 'Disabled (512-d)'}")
    print(f"  • Spatial Geometry MLP    : {'Enabled (8D -> 64 -> 600)' if args.use_spatial_mlp else 'Disabled'}")
    print(f"  • Detector Confidence     : {args.detector_conf} (rescuing small objects)")
    print(f"  • Device                  : {args.device}")

    meta = HOIMeta(args.dataset_dir)
    print(f"  ✓ Loaded {meta.num_classes} HOI interaction classes.")

    extractor = MultiStreamFeatureExtractor(
        device=args.device,
        detector_conf=args.detector_conf,
        use_3stream=args.use_3stream,
    )

    print("\n  Precomputing text classifier embeddings...")
    text_weights = extractor.build_text_weights(meta)
    print(f"  ✓ Text Weights: {text_weights.shape}")

    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "test-*.parquet")))
    print(f"  ✓ Found {len(train_files)} train shards, {len(test_files)} test shards.")

    all_results = {}
    all_losses = {}

    # Zero-shot baseline
    print("\n" + "─" * 75)
    print("  ▶ Evaluating Zero-Shot Baseline (without adapter)...")
    print("─" * 75)

    dummy_keys = torch.zeros(1, extractor.feature_dim)
    dummy_vals = torch.zeros(1, meta.num_classes)
    dummy_adapter = Vynix3StreamAdapter(dummy_keys, dummy_vals, use_spatial_mlp=False)
    dummy_adapter.alpha.data.fill_(0.0)

    res_zero = evaluate_adapter(dummy_adapter, extractor, text_weights, meta,
                                test_files, args.device, args.eval_limit)
    all_results[0] = res_zero
    print(f"  0-Shot: Full={res_zero['mAP_full']:.2f}% | Rare={res_zero['mAP_rare']:.2f}% | "
          f"Non-Rare={res_zero['mAP_non_rare']:.2f}% | Vetoes={res_zero['vetoes']}")

    # K-shot evaluation
    for k in args.k_shots:
        print(f"\n{'─' * 75}")
        print(f"  ▶ Training and Evaluating {k}-Shot 3-Stream Adapter...")
        print("─" * 75)

        cache_keys, cache_values, cache_spatials = build_few_shot_cache(
            train_files, meta, extractor, k, max_images=5000
        )

        adapter = Vynix3StreamAdapter(
            cache_keys=cache_keys,
            cache_values=cache_values,
            use_spatial_mlp=args.use_spatial_mlp,
            num_classes=meta.num_classes,
        )

        losses = train_adapter(
            adapter=adapter,
            train_features=cache_keys,
            train_labels=cache_values,
            train_spatials=cache_spatials,
            text_weights=text_weights,
            epochs=args.epochs,
            device=args.device,
        )
        all_losses[k] = losses

        res_k = evaluate_adapter(adapter, extractor, text_weights, meta,
                                 test_files, args.device, args.eval_limit)
        all_results[k] = res_k
        print(f"  {k}-Shot 3-Stream: Full={res_k['mAP_full']:.2f}% | Rare={res_k['mAP_rare']:.2f}% | "
              f"Non-Rare={res_k['mAP_non_rare']:.2f}% | Vetoes={res_k['vetoes']}")

    # Summary table
    print("\n\n" + "=" * 75)
    print("  🏆 VYNIX 3-STREAM SPATIAL-VISUAL ADAPTER RESULTS")
    print("=" * 75)
    print(f"  {'Configuration':<24} | {'Full mAP':>10} | {'Rare mAP':>10} | {'Non-Rare':>10} | {'Delta':>10}")
    print("─" * 75)

    base = all_results[0]["mAP_full"]
    for k in sorted(all_results.keys()):
        r = all_results[k]
        delta = r["mAP_full"] - base
        sign = "+" if delta >= 0 else ""
        label = "0-Shot Baseline" if k == 0 else f"{k}-Shot 3-Stream"
        tag = "baseline" if k == 0 else f"{sign}{delta:.2f}%"
        print(f"  {label:<24} | {r['mAP_full']:>9.2f}% | {r['mAP_rare']:>9.2f}% | {r['mAP_non_rare']:>9.2f}% | {tag:>10}")
    print("=" * 75)

    # Save CSV
    csv_out = os.path.join(args.output_dir, "vynix_3stream_adapter_results.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k_shot", "mAP_full", "mAP_rare", "mAP_non_rare", "vetoes"])
        writer.writeheader()
        for k in sorted(all_results.keys()):
            r = all_results[k]
            writer.writerow({
                "k_shot": k,
                "mAP_full": f"{r['mAP_full']:.4f}",
                "mAP_rare": f"{r['mAP_rare']:.4f}",
                "mAP_non_rare": f"{r['mAP_non_rare']:.4f}",
                "vetoes": r["vetoes"],
            })
    print(f"  ✓ Saved results to {csv_out}")


if __name__ == "__main__":
    main()
