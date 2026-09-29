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

def build_full_dataset_cache(train_files, meta, extractor, device, max_images=None):
    """
    Extracts visual features, labels, and spatial geometry from the ENTIRE training set.
    """
    print("\n  [Extraction] Loading Train Parquet files...")
    dfs = [pd.read_parquet(f) for f in train_files]
    train_df = pd.concat(dfs, ignore_index=True)
    if max_images:
        train_df = train_df.iloc[:max_images]
        print(f"  [Extraction] Limited to {max_images} images for testing.")

    n_images = len(train_df)
    print(f"  [Extraction] Indexed {n_images} training images. Starting feature extraction...")

    cache_keys_list = []
    cache_values_list = []
    spatial_vecs_list = []

    extractor.yolo.to(device)


    for img_idx in tqdm(range(n_images), desc="  Extracting Features", unit="img"):
        row = train_df.iloc[img_idx]
        pos_str = str(row.get("positive_objects", ""))
        if not pos_str.strip() or pos_str.lower() == "nan":
            continue

        pos_list = parse_int_list(pos_str)
        pos_set = set(pos_list)

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

        # For every ground truth HOI, try to find a matching (person, object) bounding box pair
        for (p_box, p_conf) in persons:
            for (o_box, o_conf, o_cls) in objects:
                coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
                if not coco_name:
                    continue
                hico_name = normalize_name(coco_name)

                entries = meta.obj_to_entries.get(hico_name, [])
                if not entries:
                    continue

                # Check if this object interaction is in the ground truth
                rel_ids = [hid for _, _, hid in entries if hid in pos_set]
                if not rel_ids:
                    continue

                u_box = compute_union_box(p_box, o_box)
                f_fused = extractor.extract_visual_feature(pil_img, p_box, o_box, u_box)
                s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)

                label = torch.zeros(meta.num_classes)
                for hid in rel_ids:
                    label[hid] = 1.0

                cache_keys_list.append(f_fused.cpu())
                cache_values_list.append(label.cpu())
                spatial_vecs_list.append(s_vec.cpu())

    if not cache_keys_list:
        raise ValueError("Failed to extract any features. Check dataset or YOLO confidence.")

    cache_keys = torch.cat([k.view(1, -1) for k in cache_keys_list], dim=0)    # (M, 1536)
    cache_values = torch.stack(cache_values_list)     # (M, 600)
    cache_spatials = torch.stack(spatial_vecs_list)   # (M, 8)

    # L2 Normalize keys
    cache_keys = nn.functional.normalize(cache_keys, dim=1)

    print(f"  ✓ Full Cache built: {cache_keys.shape[0]} entries × {cache_keys.shape[1]}-d visual features")
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("===========================================================================")
    print("  PROJECT VYNIX — FULL DATASET TRAINING PIPELINE")
    print("===========================================================================")

    meta = HOIMeta(args.dataset_dir)
    extractor = MultiStreamFeatureExtractor(device=args.device, detector_conf=0.08, use_3stream=True)
    text_weights = extractor.build_text_weights(meta)

    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "train-*.parquet")))
    cache_keys, cache_values, cache_spatials = build_full_dataset_cache(
        train_files, meta, extractor, args.device, max_images=args.limit
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
