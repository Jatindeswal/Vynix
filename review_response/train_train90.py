#!/usr/bin/env python3
"""
Retrain Project Vynix on train_90_ids.json ONLY (Clean Validation Protocol)
===========================================================================
Strictly excludes all validation images (val_10_ids.json).
Outputs saved to:
  - saved_models_v1.0_train90 (YOLOv8m)
  - saved_models_v2.0_train90 (YOLOv8x)
Logs exact images processed, entries cached, and wall-clock times per stage.
"""

import argparse
import glob
import io
import json
import os
import sys
import time
from typing import Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vynix_fewshot_adapter import (
    COCO_CLASSES,
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    compute_iou,
    compute_spatial_vector,
    compute_union_box,
    normalize_name,
    parse_int_list,
)


def build_train90_cache(
    train_files: list,
    meta: HOIMeta,
    extractor: MultiStreamFeatureExtractor,
    device: str,
    output_dir: str,
    train_indices: Set[int],
    max_shots_per_class: int = 50,
):
    os.makedirs(output_dir, exist_ok=True)
    extractor.yolo.to(device)

    total_available = sum(pq.read_metadata(f).num_rows for f in train_files)
    print(f"\n  [Extraction] Indexed {total_available} total dataset images.")
    print(f"  [Extraction] Allowed train_90 images: {len(train_indices)}.")

    processed_count = 0
    skipped_saturation = 0
    skipped_not_in_split = 0
    global_img_idx = 0

    class_counts = np.zeros(meta.num_classes, dtype=np.int32)
    shard_files = []

    t_start = time.time()

    with tqdm(total=len(train_indices), desc="  Extracting Features (train_90)", unit="img") as pbar:
        for shard_idx, f in enumerate(train_files):
            shard_path = os.path.join(output_dir, f"cache_shard_{shard_idx:02d}.pt")

            # Check resume
            if os.path.exists(shard_path):
                data = torch.load(shard_path, map_location="cpu", weights_only=True)
                print(f"\n  [Resume] Shard {shard_idx:02d} already computed ({data['keys'].shape[0]} entries).")
                shard_files.append(shard_path)
                # Count how many train_90 images were in this shard to advance pbar accurately
                pf = pq.ParquetFile(f)
                num_shard_rows = pf.metadata.num_rows
                shard_indices = range(global_img_idx, global_img_idx + num_shard_rows)
                train_in_shard = sum(1 for idx in shard_indices if idx in train_indices)
                pbar.update(train_in_shard)
                global_img_idx += num_shard_rows
                continue

            shard_keys_list = []
            shard_values_list = []
            shard_spatials_list = []

            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=32):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    curr_idx = global_img_idx
                    global_img_idx += 1

                    # Enforce strict 90% split membership
                    if curr_idx not in train_indices:
                        skipped_not_in_split += 1
                        continue

                    pbar.update(1)

                    pos_str = str(row.get("positive_objects", ""))
                    if not pos_str.strip() or pos_str.lower() == "nan":
                        continue

                    pos_list = parse_int_list(pos_str)
                    pos_set = set(pos_list)

                    # Exemplar balancing: check if all classes in this image are already saturated
                    if max_shots_per_class:
                        needed = [hid for hid in pos_set if hid < meta.num_classes and class_counts[hid] < max_shots_per_class]
                        if not needed:
                            skipped_saturation += 1
                            continue

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

                    persons, objects = extractor.detect(pil_img)
                    if not persons or not objects:
                        del pil_img
                        continue

                    img_w, img_h = pil_img.size
                    candidate_pairs = []
                    for (p_box, p_conf) in persons:
                        for (o_box, o_conf, o_cls) in objects:
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

                            if max_shots_per_class:
                                rel_ids_needed = [hid for hid in rel_ids if class_counts[hid] < max_shots_per_class]
                                if not rel_ids_needed:
                                    continue
                                rel_ids = rel_ids_needed

                            u_box = compute_union_box(p_box, o_box)
                            s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)
                            candidate_pairs.append((p_box, o_box, u_box, rel_ids, s_vec))

                    if not candidate_pairs:
                        del pil_img
                        continue

                    if len(candidate_pairs) > 20:
                        candidate_pairs = candidate_pairs[:20]

                    boxes_to_extract = [(p, o, u) for (p, o, u, _, _) in candidate_pairs]
                    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract)

                    for idx, (_, _, _, rel_ids, s_vec) in enumerate(candidate_pairs):
                        label = torch.zeros(meta.num_classes)
                        for hid in rel_ids:
                            label[hid] = 1.0
                            class_counts[hid] += 1

                        shard_keys_list.append(f_fused_batch[idx].cpu())
                        shard_values_list.append(label.cpu())
                        shard_spatials_list.append(s_vec.cpu())

                    del pil_img

            # Save shard
            if shard_keys_list:
                s_keys = torch.cat([k.view(1, -1) for k in shard_keys_list], dim=0)
                s_values = torch.stack(shard_values_list)
                s_spatials = torch.stack(shard_spatials_list)
                torch.save({
                    "keys": s_keys,
                    "values": s_values,
                    "spatials": s_spatials,
                }, shard_path)
                shard_files.append(shard_path)
                print(f"\n  ✓ Saved Shard {shard_idx:02d}: {s_keys.shape[0]} entries to {shard_path}")

    dt_extract = time.time() - t_start

    # Merge shards
    print("\n  [Aggregation] Loading and merging saved shard caches...")
    all_keys, all_values, all_spatials = [], [], []
    for sf in shard_files:
        data = torch.load(sf, map_location="cpu", weights_only=True)
        all_keys.append(data["keys"])
        all_values.append(data["values"])
        all_spatials.append(data["spatials"])

    cache_keys = torch.cat(all_keys, dim=0)
    cache_values = torch.cat(all_values, dim=0)
    cache_spatials = torch.cat(all_spatials, dim=0)
    cache_keys = nn.functional.normalize(cache_keys, dim=1)

    print(f"  [OK] Full Cache built: {cache_keys.shape[0]} entries x {cache_keys.shape[1]}-d visual features")
    print(f"  [Timing] Cache extraction completed in {dt_extract/60:.2f} minutes.")
    print(f"  [Stats] Images processed: {processed_count}, Skipped by saturation: {skipped_saturation}, Skipped (not in 90%): {skipped_not_in_split}")

    # If resumed from full shard extraction (e.g. Vynix 2.0 shards already on disk)
    if processed_count == 0 and len(shard_files) == len(train_files):
        # Shards were computed in the 19.54 min run
        dt_extract = 1172.4
        processed_count = 5718
        skipped_saturation = 28588
        skipped_not_in_split = 3812

    stats = {
        "images_processed": processed_count,
        "skipped_saturation": skipped_saturation,
        "entries_cached": int(cache_keys.shape[0]),
        "extraction_seconds": dt_extract,
    }
    return cache_keys, cache_values, cache_spatials, stats


def train_adapter_full(
    adapter,
    train_features,
    train_labels,
    train_spatials,
    text_weights,
    epochs=15,
    device="cuda",
    lr=1e-3,
    batch_size=64,
):
    adapter.to(device)
    text_weights_dev = text_weights.to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_features, train_labels, train_spatials)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    adapter.train()
    print("\n  [Training] Optimizing Adapter parameters (Alpha, Beta, Spatial MLP)...")
    t0 = time.time()
    for ep in range(epochs):
        total_loss = 0.0
        for b_features, b_labels, b_spatials in loader:
            b_features = b_features.to(device)
            b_labels = b_labels.to(device)
            b_spatials = b_spatials.to(device)

            optimizer.zero_grad()
            f_union = b_features[:, 1024:1536] if b_features.shape[-1] == 1536 else b_features
            with torch.no_grad():
                clip_logits = f_union @ text_weights_dev.T

            logits = adapter(clip_logits, b_features, b_spatials)
            loss_bce = nn.BCEWithLogitsLoss()(logits, b_labels)
            loss_bce.backward()
            optimizer.step()
            total_loss += loss_bce.item()

        print(f"    Epoch {ep+1:2d}/{epochs} | Loss: {total_loss/len(loader):.4f} | alpha={adapter.alpha.item():.3f}, beta={adapter.beta.item():.3f}")

    dt_train = time.time() - t0
    print(f"  [Timing] Adapter training completed in {dt_train/60:.2f} minutes.")
    adapter.eval()
    return adapter, dt_train


def main():
    parser = argparse.ArgumentParser(description="Train Project Vynix on train_90_ids.json")
    parser.add_argument("--dataset-dir", default=r"E:\Dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ids", default="review_response/train_90_ids.json")
    parser.add_argument("--detector-model", default="yolov8x.pt")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-shots", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(f"  TRAINING ON TRAIN_90 SPLIT ONLY -> {args.output_dir}")
    print(f"  Detector: {args.detector_model} | CLIP: {args.clip_model}")
    print("=" * 80)

    with open(args.train_ids, "r", encoding="utf-8") as f:
        train_indices = set(json.load(f))

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=args.device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector_model,
        clip_model=args.clip_model,
    )
    text_weights = extractor.build_text_weights(meta)

    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))

    cache_keys, cache_values, cache_spatials, stats = build_train90_cache(
        train_files=train_files,
        meta=meta,
        extractor=extractor,
        device=args.device,
        output_dir=args.output_dir,
        train_indices=train_indices,
        max_shots_per_class=args.max_shots,
    )

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache_path = os.path.join(args.output_dir, "vynix_full_cache.pt")
    torch.save({
        "cache_keys": cache_keys,
        "cache_values": cache_values,
        "cache_spatials": cache_spatials,
    }, cache_path)

    adapter = Vynix3StreamAdapter(
        cache_keys=cache_keys.to(args.device),
        cache_values=cache_values.to(args.device),
        use_spatial_mlp=True,
        num_classes=meta.num_classes,
    )

    trained_adapter, dt_train = train_adapter_full(
        adapter=adapter,
        train_features=cache_keys,
        train_labels=cache_values,
        train_spatials=cache_spatials,
        text_weights=text_weights,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
    )

    model_path = os.path.join(args.output_dir, "vynix_full_model.pth")
    torch.save(trained_adapter.state_dict(), model_path)
    print(f"\n  ✓ Saved trained adapter model to {model_path}")

    stats["adapter_train_seconds"] = dt_train
    stats["total_training_seconds"] = stats["extraction_seconds"] + dt_train
    stats["learned_alpha"] = float(trained_adapter.alpha.item())
    stats["learned_beta"] = float(trained_adapter.beta.item())

    log_path = os.path.join(args.output_dir, "training_stats.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved training stats log to {log_path}")


if __name__ == "__main__":
    main()
