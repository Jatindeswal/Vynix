# Cache & Exemplar Audit (Answers Requirement 5)

## 1. Measured Cache Entries vs. Image Capping

In `train_vynix_full.py:80-84`, `build_full_dataset_cache` checks whether all positive HOI classes present in an image have already reached the exemplar cap (`--max-shots 50`, default):
```python
needed = [hid for hid in pos_set if hid < meta.num_classes and class_counts[hid] < max_shots_per_class]
if not needed:
    continue
```
If an image contains only saturated head classes, **both YOLO detector inference and 3-stream CLIP crop extraction are bypassed**.

### Full-Dataset Original Runs (`saved_models_v1.0` and `saved_models_v2.0`)
Inspection of the saved cache files (`saved_models_v1.0/vynix_full_cache.pt` and `saved_models_v2.0/vynix_full_cache.pt`) confirms that the `.pt` files store exclusively:
- `cache_keys`: `(N, 1536)` float tensor
- `cache_values`: `(N, 600)` one-hot float tensor
- `cache_spatials`: `(N, 8)` spatial geometry float tensor

**Unique image IDs are not stored in the cache shards or full cache files.** Because raw stdout logs from the original full-train runs were not preserved with per-image logs, the exact number of unique training images processed vs. skipped on the full 38,118 image training set is **`[UNKNOWN]`** and must be provided by the original authors if required. The exact number of cached exemplar pairs is directly measured:

| Quantity | Full Train Vynix 1.0 (`yolov8m.pt`) | Full Train Vynix 2.0 (`yolov8x.pt`) | Tag |
|---|---|---|---|
| **Total Available Dataset Images** | `38,118` | `38,118` | `[MEASURED]` |
| **Cached Exemplar Pairs** | **19,824** | **20,006** | `[MEASURED]` |
| **Unique Images Processed** | `[UNKNOWN]` (image IDs not saved in `.pt`) | `[UNKNOWN]` (image IDs not saved in `.pt`) | `[UNKNOWN]` |
| **Unique Images Skipped** | `[UNKNOWN]` | `[UNKNOWN]` | `[UNKNOWN]` |

---

### Clean Split Retrained Runs (`train_90_ids.json`, 34,306 images)
In the clean retraining protocol (`review_response/train_train90.py`), image counters were logged directly to `training_stats.json`:

| Quantity | Vynix 1.0 (`saved_models_v1.0_train90`) | Vynix 2.0 (`saved_models_v2.0_train90`) | Tag |
|---|---|---|---|
| **Allowed Images (`train_90_ids.json`)** | `34,306` | `34,306` | `[MEASURED]` |
| **Images Actually Processed** | **5,927** (17.3%) | **5,718** (16.7%) | `[MEASURED]` |
| **Images Skipped by Saturation** | **28,379** (82.7%) | **28,588** (83.3%) | `[MEASURED]` |
| **Excluded Validation Images** | `3,812` (100% bypassed) | `3,812` (100% bypassed) | `[MEASURED]` |
| **Cached Exemplar Pairs** | **19,579** | **19,673** | `[MEASURED]` |
| **Extraction Time** | **15.91 min** (954.8 s) | **19.54 min** (1,172.4 s) | `[MEASURED]` |
| **Adapter Training Time** | **0.57 min** (34.3 s) | **1.49 min** (89.3 s) | `[MEASURED]` |
| **Total Stage Training Time** | **16.48 min** | **21.03 min** | `[MEASURED]` |
| **Learned Parameters** | $\alpha=0.955, \beta=5.558$ | $\alpha=0.940, \beta=5.592$ | `[MEASURED]` |

This confirms why training takes only 16–21 minutes on an RTX 3050 Ti: once frequent HOI categories saturate at 50 exemplars, over 82% of dataset images are skipped entirely without running YOLO or CLIP.
