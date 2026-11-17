# Methodological Disclosures and Hyperparameter Provenance

This document records all methodological decisions, exploratory runs, and design choices made prior to the establishment of the frozen validation protocol on `val_10`. Every item is documented transparently in accordance with peer-review standards.

---

## 1. Exploratory Test-Set Evaluations (`eval_limit=300`)
- **[LOG] `vynix_fewshot_adapter.py` lines 767–769 (`eval_limit: Optional[int] = None`)**:
  During early system development, interactive test subsets (e.g., `eval_limit=300` on `test-*.parquet`) were used to benchmark latency and verify pipeline integration on a single laptop GPU (RTX 3050 Ti).
- **Impact**: Preliminary values of $\lambda=0.90$ (detector power weighting) and $\gamma_{\text{obj}}=0.10$ (object semantic gate exponent) were initially explored against test subsets before formal coordinate-wise tuning was conducted on the held-out `val_10` split.

---

## 2. Text Prompt Ensembling
- **[LOG] `vynix_fewshot_adapter.py` lines 122–155 (`HOI_TEMPLATES`, `NO_INT_TEMPLATES`)**:
  The zero-shot text classification weights were expanded from a single template (`a photo of a person {gerund} a {object}`) to an 8-template prompt ensemble for positive interactions and 4 templates for negative interactions.
- **Provenance**: The template structure was refined based on observing interaction disambiguation on test examples.

---

## 3. Historical Geometric Veto Parameter ($\sigma=0.08$)
- **[LOG] Original paper manuscript**:
  The soft veto scale parameter $\sigma=0.08$ reported in the paper was originally selected following a preliminary sweep across the test partition ($\sigma \in \{0.04, 0.06, 0.08, 0.10, 0.12, 0.16\}$) where $\sigma=0.08$ was selected.
- **Validation Audit**:
  Under the strictly held-out `val_10` protocol with Default box-level mAP, the grid was extended to $\sigma \in \{0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16\}$. The objective function favored smaller $\sigma$, with $\sigma=0.01$ yielding the highest Full mAP ($15.06\%$ on v1.0, $16.81\%$ on v2.0).

---

## 4. Detector Proposal Confidence Threshold ($\tau_{\text{det}}$) and Inactive Sweep
- **[LOG] `vynix_fewshot_adapter.py` lines 281, 414–429 (`detect`)**:
  The paper manuscript claimed candidate pairs were filtered at threshold $\tau_{\text{det}} = 0.08$.
- **Audit Finding [MEASURED]**:
  In `MultiStreamFeatureExtractor.detect()`, `self.yolo(pil_image)` was invoked without passing `conf=self.detector_conf`. In the Ultralytics YOLO framework, `model.__call__` has a default keyword argument `conf=0.25`. Consequently, all detected bounding boxes from YOLOv8 had confidence $\ge 0.25$ before reaching the `if conf < self.detector_conf:` check.
- **Consequence on Hyperparameter Sweeps [MEASURED]**:
  The planned sweep over detector threshold $\tau_{\text{det}} \in \{0.05, 0.08, 0.12\}$ was mechanically inactive on the cached proposals because the minimum person confidence in the cache was $0.25006$ and minimum object confidence was $0.25001$. As measured on `val_10`, the number of candidate pairs was completely invariant across all three thresholds: exactly $64,344$ pairs ($16.88$ pairs/image) for Vynix 1.0 and $66,685$ pairs ($17.49$ pairs/image) for Vynix 2.0. By the pre-committed tie-break rule, the nominal default $\tau_{\text{det}} = 0.08$ was retained.

---

## 5. Cache Saturation and Effective Training Images (50-Shot Cap, Not 16-Shot)
- **Code & Log Evidence [LOG]**:
  - `train_vynix_full.py:24`: `def build_full_dataset_cache(..., max_shots_per_class=50)`
  - `train_vynix_full.py:247`: `parser.add_argument("--max-shots", type=int, default=50)`
  - `vynix_cli.py:468, 689`: `default=50`
  - `process.md:136`: *"Class-balanced prototype capping (`max_shots_per_class=50`): images with only saturated head categories bypass CLIP feature extraction. This reduced feature extraction time from ~5 hours to 17.33 minutes."*
  - `review_response/train_train90.py:50`: `max_shots_per_class: int = 50`
- **Audit Resolution [MEASURED]**:
  The mention of "16-shot" in early discussions arose from informal analogy to few-shot vision-language literature (e.g., Tip-Adapter 16-shot benchmark). In Project Vynix's actual code, configs, and execution logs, the exemplar cap is strictly **50-shot per HOI category** (`max_shots_per_class=50`).
- **Empirical Impact [MEASURED]**:
  Although HICO-DET contains 38,118 training images, the 50-shot cap caused the cache extraction loop to bypass $\sim 28,400$ training images once frequent categories reached 50 positive exemplars. Specifically on `train_90` (34,306 images):
  - **Vynix 1.0**: Exactly 5,927 images processed, 28,379 images bypassed, 19,579 pairs cached in 15.91 minutes.
  - **Vynix 2.0**: Exactly 5,718 images processed, 28,588 images bypassed, 19,673 pairs cached in 19.54 minutes.
  At test time, zero images are bypassed: all 9,658 test images are fully evaluated.
