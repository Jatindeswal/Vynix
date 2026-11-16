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

## 4. Detector Proposal Confidence Threshold ($\tau_{\text{det}}$)
- **[LOG] `vynix_fewshot_adapter.py` lines 281, 414–429 (`detect`)**:
  The paper manuscript claimed candidate pairs were filtered at threshold $\tau_{\text{det}} = 0.08$.
- **Audit Finding [MEASURED]**:
  In `MultiStreamFeatureExtractor.detect()`, `self.yolo(pil_image)` was invoked without passing `conf=self.detector_conf`. In the Ultralytics YOLO framework, `model.__call__` has a default keyword argument `conf=0.25`. Consequently, all detected bounding boxes from YOLOv8 had confidence $\ge 0.25$ before reaching the `if conf < self.detector_conf:` check.
- **Consequence**: The effective proposal threshold throughout the original paper experiments, cache generation, and test evaluation was $\tau_{\text{det}} \ge 0.25$, yielding an average of $\sim 16.9$ to $17.5$ candidate pairs per image.

---

## 5. Cache Saturation and Effective Training Images
- **[MEASURED] Train-90 audit and full-train log verification**:
  Although HICO-DET contains 38,118 training images, the 16-shot class saturation cap (`max_shots=16`) caused the cache extraction loop to bypass $\sim 28,400$ images once all 600 HOI categories reached quota. The full training cache was constructed from $\sim 5,700$ to $5,900$ images containing $\sim 19,600$ pairs.
