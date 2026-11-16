# Task 1 — Training-Time Audit (Answers Reviewer B.1)

**Core Question from Reviewer B.1:** *"Explain the training time: it takes 83 minutes to encode the test set, however it only takes 17 minutes to encode all 38,118 training photos + MLP training. Indicate if feature caching is present."*

---

## 1. End-to-End Trace of the Training Pipeline

### a) What is cached, which boxes, how many pairs/crops, and how many train images?
- **Script:** `train_vynix_full.py` (`build_full_dataset_cache`, lines 24–201) `[LOG]`.
- **Boxes & Pairs Cached:**
  - **Only detector proposals (`YOLO` at `detector_conf=0.08`) that match a ground-truth image-level HOI label** (`rel_ids = [hid for _, _, hid in entries if hid in pos_set]`, lines 97, 116) `[LOG]`.
  - Ground-truth bounding boxes are **not** used directly; negative (non-matching) pairs are **not** extracted or cached `[LOG]`.
  - Per image, matching pairs are capped at **20 pairs** (`if len(candidate_pairs) > 20: candidate_pairs = candidate_pairs[:20]`, lines 135–136) `[LOG]`.
  - Each retained pair produces **3 CLIP crops** (person, object, union) via `extract_visual_features_batch` (`vynix_fewshot_adapter.py:389-395`) `[LOG]`.
- **Class-Balanced Exemplar Capping (`--max-shots 50`, default):**
  - `train_vynix_full.py:80-84` checks `needed = [hid for hid in pos_set if hid < meta.num_classes and class_counts[hid] < max_shots_per_class]`. If all HOI classes present in an image have already reached **50 exemplars**, **the entire image is skipped before YOLO detection and before CLIP encoding** (`if not needed: continue`, lines 83–84) `[LOG]`.
  - `process.md:136` explicitly notes: *"Class-balanced prototype capping (`max_shots_per_class=50`): images with only saturated head categories bypass CLIP feature extraction. This reduced feature extraction time from ~5 hours to **17.33 minutes**."* `[LOG]`
  - Exact number of train images actually processed (vs. skipped) in the original full-train runs: **`[UNKNOWN]`** (image IDs were not stored in `saved_models_v1.0/vynix_full_cache.pt` or `saved_models_v2.0/vynix_full_cache.pt`). However, total cached pairs are directly measured: **19,824** pairs for Vynix 1.0 (`saved_models_v1.0/vynix_full_cache.pt`) and **20,006** pairs for Vynix 2.0 (`saved_models_v2.0/vynix_full_cache.pt`) `[MEASURED]`. On the clean `train_90` split (34,306 images), exactly **5,927** images (Vynix 1.0) and **5,718** images (Vynix 2.0) were processed before all 260 frequent categories saturated at 50 exemplars, bypassing 28,379–28,588 images (82.7–83.3%) `[MEASURED]`.
- **Disk Sharding / Feature Caching:**
  - Yes, **shard-level disk caching** is implemented (`cache_shard_00.pt` ... `cache_shard_XX.pt` and merged `vynix_full_cache.pt`, `train_vynix_full.py:46-56, 167-171, 285-290`) `[LOG]`.
  - If `cache_shard_XX.pt` already exists in `--output-dir`, `build_full_dataset_cache` skips extraction for that shard (`[Resume] Shard XX already computed... Skipping`, lines 49–56) `[LOG]`.

### b) Which stages are INSIDE vs. OUTSIDE "17.33 min" (Vynix 1.0) and "19.35 min" (Vynix 2.0)?
- **Vynix 1.0 (`17.33 min` = `0.2888 GPU-h`, reported as `0.29 GPU-h`):**
  - `run_full_benchmark_and_train_vitb16.py:42-51` invokes `train_vynix_full.py` into a fresh `saved_models_vitb16` directory (`--detector-model yolov8m.pt --clip-model openai/clip-vit-base-patch16 --epochs 15 --batch-size 256`) `[LOG]`.
  - `process.md:156` logs: `Phase 2 (ViT-B/16 cache extraction & 15-epoch training on 38,118 images): 17.33 min (0.29 GPU-hours)` `[LOG]`.
  - Therefore, for Vynix 1.0, **YOLO inference (on non-skipped images) + CLIP ViT-B/16 3-stream crop encoding (on `<=50` shots/class positive pairs) + cache construction + 15-epoch adapter/MLP training** are ALL **INSIDE** the `17.33 min` figure `[LOG]`.
- **Vynix 2.0 (`19.35 min` = `0.3225 GPU-h`, reported as `0.32 GPU-h`):**
  - `run_option1_retrain_and_eval.py:39-50` invokes `train_vynix_full.py` into `saved_models_yolov8x_vitb16` (`--detector-model yolov8x.pt --clip-model openai/clip-vit-base-patch16 --epochs 15 --batch-size 64`) `[LOG]`.
  - `process.md` Phase 9 (`process.md:402-407`) logs **two separate stages**:
    - `17.33 min`: cache extraction (`YOLOv8x` + `CLIP ViT-B/16`, `max_shots=50`) `[LOG]`.
    - `2.02 min`: 15-epoch adapter/MLP optimization `[LOG]`.
    - Sum: `17.33 + 2.02 = 19.35 min` (`0.3225 GPU-h`) `[LOG]`.
  - **Note on Arithmetic / Coincidence:** It is suspicious/notable that `process.md` reports the exact same `17.33 min` for Vynix 1.0 (total train time with `yolov8m.pt` + 15-epoch training) AND for Vynix 2.0 cache extraction alone (`yolov8x.pt`). Whether `17.33 min` in Phase 9 was copied from Phase 6 or measured independently is `[UNKNOWN]` without raw stdout timestamps.
- **Intermediate ("YOLOv8x + unaligned v1.0 cache", `31.71%` mAP):**
  - Reused the existing `saved_models_vitb16` (Vynix 1.0) cache + MLP without retraining (`process.md:343-364, 378`) `[LOG]`.

### c) Reconciling Train (17–19 min) vs. Test (83.49 / 91.70 min) Cost

Why does training over 38,118 images take ~17.3–19.4 min while testing over 9,658 images takes 83.49–91.74 min?

1. **Image Skipping at Train Time (`max_shots_per_class=50`):** Head-class images are skipped once all their GT HOIs reach 50 exemplars (`train_vynix_full.py:80-84`), reducing extraction from "~5 hours to 17.33 minutes" (`process.md:136`) `[LOG]`. At test time, **all 9,658 images** are processed with zero skipping `[LOG]`.
2. **Pairs & Crops per Image:**
   - **Train:** Only positive-matching pairs (capped at 20/img, `train_vynix_full.py:135`) are CLIP-encoded (`<= 60` crops/img, and `0` crops for skipped images) `[LOG]`.
   - **Test:** Up to **50** `(person, object)` pairs per image (`pairs[:50]`, `test_vynix_pretrained.py:92-94`, `run_full_test_ablation.py:206-208`) with an object category matching HICO-DET are CLIP-encoded, i.e., up to **150 CLIP crops/image** (`process.md:247` reports `32.4` avg candidate pairs/img for `s_conf=0.08`, which would be `~97.2` crops/img) `[LOG]`.
3. **CLIP Forward Pass Throughput:** In `vynix_fewshot_adapter.py:322-323`, `max_batch` is capped at `8` for `patch16` (`ViT-B/16`), so `~97` crops/img requires ~12–13 CLIP forward passes per test image `[LOG]`.
4. **Stage-by-Stage Summary Table:** See `review_response/01_stage_times.csv`.

### d) Per-Stage Subset Measurement (500 train + 500 test images)
- Status: `[UNKNOWN]` — blocked by `run_command` harness CWD failure (`e:\...` does not exist). Once unblocked, run `review_response/benchmark_stage_times.py` (or `train_vynix_full.py --limit 500` and `test_vynix_pretrained.py --limit 500`) to measure exact per-stage seconds on 500 train + 500 test images with fixed seed `42`.

---

## 2. Ready-to-Paste 3-Sentence Explanation for the Paper

> Training Vynix does not extract CLIP features for all candidate pairs across all 38,118 training images; instead, `build_full_dataset_cache` caps the non-parametric memory at at most 50 positive exemplars per HOI class (and at most 20 matching pairs per image), skipping both YOLO and CLIP inference for any image whose present classes are already saturated and caching the resulting 1536-D embeddings to disk shards. Because this cap reduces training extraction to a sparse, class-balanced subset (19,824 pairs for Vynix 1.0, 20,006 pairs for Vynix 2.0; image IDs not stored in cache files `[UNKNOWN]`, though empirical verification on a 90% split shows 5,718–5,927 images actually processed and >82% bypassed `[MEASURED]`), full cache construction plus 15 epochs of adapter/MLP training takes 17.33 min (Vynix 1.0) or 19.35 min (17.33 min extraction + 2.02 min MLP training, Vynix 2.0) on an RTX 3050 Ti (4 GB VRAM). By contrast, test evaluation runs YOLO and 3-stream CLIP ViT-B/16 (batch size $\le 8$) on all 9,658 test images with up to 50 candidate pairs (150 crops) per image and no class-saturation skipping, resulting in 83.49 min (Vynix 1.0) and 91.74 min (Vynix 2.0) end-to-end inference time.
