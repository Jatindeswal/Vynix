import os
import argparse
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import io
from PIL import Image

# Import existing core components
from vynix_fewshot_adapter import (
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    compute_iou,
    compute_union_box,
    compute_spatial_vector,
    normalize_name,
    COCO_CLASSES,
    parse_int_list
)

def build_full_dataset_cache(train_files, meta, extractor, device, output_dir="saved_models", max_images=None, max_shots_per_class=50):
    """
    Extracts visual features, labels, and spatial geometry from the training set via incremental shard streaming.
    Saves each shard directly to disk (cache_shard_XX.pt) to guarantee zero RAM leaks.
    """
    import pyarrow.parquet as pq
    import gc
    import numpy as np

    total_available = sum(pq.read_metadata(f).num_rows for f in train_files)
    n_target = min(total_available, max_images) if max_images else total_available
    print(f"\n  [Extraction] Indexed {total_available} images across {len(train_files)} shards. Target: {n_target} images.")

    os.makedirs(output_dir, exist_ok=True)
    extractor.yolo.to(device)
    processed_count = 0
    class_counts = np.zeros(meta.num_classes, dtype=np.int32)

    shard_files = []

    with tqdm(total=n_target, desc="  Extracting Features", unit="img") as pbar:
        for shard_idx, f in enumerate(train_files):
            shard_path = os.path.join(output_dir, f"cache_shard_{shard_idx:02d}.pt")

            # Resume existing shard if already computed
            if os.path.exists(shard_path):
                data = torch.load(shard_path, map_location="cpu", weights_only=True)
                print(f"\n  [Resume] Shard {shard_idx:02d} already computed ({data['keys'].shape[0]} entries). Skipping.")
                shard_files.append(shard_path)
                num_shard_rows = pq.read_metadata(f).num_rows
                pbar.update(num_shard_rows)
                processed_count += num_shard_rows
                continue

            shard_keys_list = []
            shard_values_list = []
            shard_spatials_list = []

            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                if max_images and processed_count >= max_images:
                    break
                processed_count += 1
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
                        continue

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

                        # Check exemplar budget
                        if max_shots_per_class:
                            rel_ids_needed = [hid for hid in rel_ids if class_counts[hid] < max_shots_per_class]
                            if not rel_ids_needed:
                                continue
                            rel_ids = rel_ids_needed

                        u_box = compute_union_box(p_box, o_box)
                        s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)
                        candidate_pairs.append((p_box, o_box, u_box, rel_ids, s_vec))

                if not candidate_pairs:
                    continue

                if len(candidate_pairs) > 20:
                    candidate_pairs = candidate_pairs[:20]

                # Batched visual feature extraction for all positive pairs in this image
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

            del df
            gc.collect()

            # Save this shard immediately to disk to free RAM
            if shard_keys_list:
                s_keys = torch.cat([k.view(1, -1) for k in shard_keys_list], dim=0)
                s_values = torch.stack(shard_values_list)
                s_spatials = torch.stack(shard_spatials_list)
                torch.save({
                    "keys": s_keys,
                    "values": s_values,
                    "spatials": s_spatials
                }, shard_path)
                shard_files.append(shard_path)
                print(f"\n  ✓ Saved Shard {shard_idx:02d}: {s_keys.shape[0]} entries to {shard_path}")

            del shard_keys_list, shard_values_list, shard_spatials_list
            gc.collect()

            if max_images and processed_count >= max_images:
                break

    if not shard_files:
        raise ValueError("Failed to extract any features. Check dataset or YOLO confidence.")

    # Combine all saved shards
    print("\n  [Aggregation] Loading and merging saved shard caches...")
    all_keys = []
    all_values = []
    all_spatials = []
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
    return cache_keys, cache_values, cache_spatials

def train_adapter_full(adapter, train_features, train_labels, train_spatials, text_weights, epochs=20, device="cuda", lr=1e-3, batch_size=256):
    adapter.to(device)
    text_weights_dev = text_weights.to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_features, train_labels, train_spatials)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    adapter.train()
    print("\n  [Training] Optimizing Adapter parameters (Alpha, Beta, Spatial MLP)...")
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

    return adapter

def main():
    parser = argparse.ArgumentParser(description="Train Project Vynix on Full Dataset")
    parser.add_argument("--dataset-dir", type=str, default="/app")
    parser.add_argument("--output-dir", type=str, default="saved_models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit training images for testing")
    parser.add_argument("--max-shots", type=int, default=50, help="Max exemplars per class (0 for unlimited)")
    parser.add_argument("--detector-model", type=str, default="yolov8m.pt", help="YOLO detector model (e.g. yolov8m.pt, yolov8l.pt, yolov8n.pt)")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP vision backbone (e.g. openai/clip-vit-base-patch32, openai/clip-vit-base-patch16)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("===========================================================================")
    print("  PROJECT VYNIX — FULL DATASET TRAINING PIPELINE")
    print("===========================================================================")
    print(f"  Detector: {args.detector_model}")
    print(f"  CLIP Backbone: {args.clip_model}")

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(
        device=args.device,
        detector_conf=0.08,
        use_3stream=True,
        detector_model=args.detector_model,
        clip_model=args.clip_model
    )
    text_weights = extractor.build_text_weights(meta)

    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))
    max_shots = args.max_shots if args.max_shots > 0 else None
    cache_keys, cache_values, cache_spatials = build_full_dataset_cache(
        train_files, meta, extractor, args.device,
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

    adapter = Vynix3StreamAdapter(
        cache_keys=cache_keys.to(args.device),
        cache_values=cache_values.to(args.device),
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
        device=args.device,
        batch_size=args.batch_size
    )

    model_path = os.path.join(args.output_dir, "vynix_full_model.pth")
    torch.save(trained_adapter.state_dict(), model_path)
    print(f"\n  ✓ Saved trained adapter model to {model_path}")

if __name__ == "__main__":
    main()
