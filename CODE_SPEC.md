# Project Vynix: Mathematical and Algorithmic Code Specification (`CODE_SPEC.md`)

This document provides a rigorous, step-by-step mathematical specification of the Vynix human-object interaction detection pipeline as implemented in source code. Every equation, tensor dimension, hyperparameter, and decision rule is cross-referenced with exact `file:line` locations in the repository and verified against saved model weights.

---

## 1. Input Data & Object Proposal Generation

### 1.1 Input Image
- Given an RGB image $I \in \mathbb{R}^{H \times W \times 3}$.

### 1.2 Object Detection & Proposal Filtering
- **Detector**: YOLOv8 (`yolov8m.pt` for Vynix 1.0; `yolov8x.pt` for Vynix 2.0).
- **Execution** (`vynix_fewshot_adapter.py:414–429`):
  $$\mathcal{D} = \text{YOLO}(I)$$
  Yields candidate person boxes $\mathcal{B}_h = \{(b_h, s_h)\}$ and object boxes $\mathcal{B}_o = \{(b_o, s_o, c_{\text{coco}})\}$, where:
  - $b = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ in pixel coordinates.
  - $s \in [0, 1]$ is the detector classification confidence score.
  - $c_{\text{coco}} \in \{0, \dots, 79\}$ is the COCO 80-class integer index ($0 = \text{person}$).
- **Detector Confidence Threshold ($\tau_{\text{det}}$)** (`vynix_fewshot_adapter.py:422`):
  Candidate detections satisfy $s_h \ge \tau_{\text{det}}$ and $s_o \ge \tau_{\text{det}}$.
  - *Empirical Note*: In the reference implementation, Ultralytics YOLO's default argument `conf=0.25` is active internally in `self.yolo.__call__`, meaning effective proposals satisfy $s \ge 0.25$ unless explicitly passed.

### 1.3 Candidate Pair Formulation & Top-50 Truncation
- Form Cartesian product $\mathcal{P} = \mathcal{B}_h \times \mathcal{B}_o$.
- Each pair $p = (b_h, s_h, b_o, s_o, c_{\text{coco}})$ is scored by joint detector confidence:
  $$S_{\text{pair}}(p) = s_h \cdot s_o \quad (\text{vynix\_fewshot\_adapter.py:799, evaluate\_test\_unified.py:274})$$
- Truncate $\mathcal{P}$ to top-$K$ pairs ($K = 50$) sorted descending by $S_{\text{pair}}$.
- Filter out pairs whose detected object class $c_{\text{coco}}$ does not map to a HICO-DET object category (`meta.obj_to_entries`).

---

## 2. Text Prompts and Text Classifier Weights

### 2.1 Multi-Template Prompt Ensembling
- **Vocabulary**: $C = 600$ HOI classes defined in `list_action.csv` (`vynix_fewshot_adapter.py:224–244`).
- **Templates** (`vynix_fewshot_adapter.py:122–154`):
  - For positive interactions (8 templates):
    1. `"a photo of a person {g} {art} {o}"`
    2. `"a photo of someone {g} {art} {o}"`
    3. `"a human {g} {art} {o}"`
    4. `"a photo of the person {g} {art} {o}"`
    5. `"a cropped photo of a person {g} {art} {o}"`
    6. `"a close-up photo of a person {g} {art} {o}"`
    7. `"an image of a person {g} {art} {o}"`
    8. `"a photo of a person actively {g} {art} {o}"`
  - For negative interactions (`no_interaction`, 4 templates):
    1. `"a person standing near {art} {o} without interacting"`
    2. `"a photo of a person and {art} {o} without interaction"`
    3. `"a person near {art} {o} not interacting"`
    4. `"a photo of a person next to {art} {o} with no interaction"`
  where `{g}` is the verb gerund (e.g. `riding`), `{o}` is the object name (e.g. `bicycle`), and `{art}` is the indefinite article (`a` or `an`).

### 2.2 Text Classifier Weight Construction
- In `MultiStreamFeatureExtractor.build_text_weights()` (`vynix_fewshot_adapter.py:432–475`):
  For each HOI class $c \in \{0, \dots, 599\}$ with $T_c$ templates:
  $$w_{c,t} = \frac{\text{CLIP}_{\text{text}}(\text{prompt}_{c,t})}{\|\text{CLIP}_{\text{text}}(\text{prompt}_{c,t})\|_2} \in \mathbb{R}^{512}$$
  $$\bar{w}_c = \frac{1}{T_c} \sum_{t=1}^{T_c} w_{c,t}, \quad W_{\text{text}}[c, :] = \frac{\bar{w}_c}{\|\bar{w}_c\|_2} \in \mathbb{R}^{512}$$
  Yields normalized classifier matrix $W_{\text{text}} \in \mathbb{R}^{600 \times 512}$.

---

## 3. Visual Feature Extraction & CLIP Logits

### 3.1 Bounding Box Crops
For each candidate pair $(b_h, b_o)$:
1. Person crop: $b_h = [x_{h1}, y_{h1}, x_{h2}, y_{h2}]$
2. Object crop: $b_o = [x_{o1}, y_{o1}, x_{o2}, y_{o2}]$
3. Union crop (`vynix_fewshot_adapter.py:165–167`):
   $$b_u = [\min(x_{h1}, x_{o1}), \, \min(y_{h1}, y_{o1}), \, \max(x_{h2}, x_{o2}), \, \max(y_{h2}, y_{o2})]$$

### 3.2 Visual Feature Encoding & Fusion
- In `extract_visual_features_batch()` (`vynix_fewshot_adapter.py:363–402`):
  Each crop is extracted from $I$, resized to $224 \times 224$, normalized, and passed through frozen CLIP ViT-B/16 image encoder:
  $$f_h = \frac{\text{CLIP}_{\text{img}}(\text{crop}_h)}{\|\text{CLIP}_{\text{img}}(\text{crop}_h)\|_2} \in \mathbb{R}^{512}, \quad f_o = \frac{\text{CLIP}_{\text{img}}(\text{crop}_o)}{\|\text{CLIP}_{\text{img}}(\text{crop}_o)\|_2} \in \mathbb{R}^{512}, \quad f_u = \frac{\text{CLIP}_{\text{img}}(\text{crop}_u)}{\|\text{CLIP}_{\text{img}}(\text{crop}_u)\|_2} \in \mathbb{R}^{512}$$
- **Fused Visual Embedding**:
  Concatenate the three unit-norm vectors and normalize by the Euclidean norm $\sqrt{\|f_h\|^2 + \|f_o\|^2 + \|f_u\|^2} = \sqrt{3}$:
  $$f_{\text{fused}} = \frac{[f_h \,\|\, f_o \,\|\, f_u]}{\|[f_h \,\|\, f_o \,\|\, f_u]\|_2} = \frac{[f_h \,\|\, f_o \,\|\, f_u]}{\sqrt{3}} \in \mathbb{R}^{1536} \quad (\text{vynix\_fewshot\_adapter.py:400–401})$$

### 3.3 Zero-Shot CLIP Logits
- In `evaluate_adapter()` (`vynix_fewshot_adapter.py:820`) and `evaluate_test_unified.py:363`:
  Zero-shot CLIP logits are computed using the union slice $f_{\text{fused}}[:, 1024:1536] = \frac{f_u}{\sqrt{3}}$:
  $$S_{\text{vlm}} = f_{\text{fused}}[:, 1024:1536] \cdot W_{\text{text}}^\top = \frac{f_u \cdot W_{\text{text}}^\top}{\sqrt{3}} \in \mathbb{R}^{600}$$
  *Implementation Detail*: There is **no explicit logit-scale multiplier** (e.g., $100.0$ or $\exp(\tau)$ is absent). The logits have natural magnitude in $[-1/\sqrt{3}, 1/\sqrt{3}] \approx [-0.577, 0.577]$.

---

## 4. Spatial Geometry Vector & Spatial Prior MLP

### 4.1 8-D Relative Spatial Vector
- In `compute_spatial_vector()` (`vynix_fewshot_adapter.py:191–217`):
  Given bounding box coordinates and image dimensions $(W, H)$:
  $$\Delta x = \frac{(x_{h1} + x_{h2}) - (x_{o1} + x_{o2})}{2W}, \quad \Delta y = \frac{(y_{h1} + y_{h2}) - (y_{o1} + y_{o2})}{2H}$$
  $$w_h = \frac{x_{h2} - x_{h1}}{W}, \quad h_h = \frac{y_{h2} - y_{h1}}{H}, \quad w_o = \frac{x_{o2} - x_{o1}}{W}, \quad h_o = \frac{y_{o2} - y_{o1}}{H}$$
  $$\text{IoU} = \frac{\text{area}(b_h \cap b_o)}{\text{area}(b_h \cup b_o)}, \quad \text{log\_area\_ratio} = \ln\left(\frac{w_h \cdot h_h}{w_o \cdot h_o}\right)$$
  Vector formulation:
  $$g = [\Delta x, \, \Delta y, \, w_h, \, h_h, \, w_o, \, h_o, \, \text{IoU}, \, \text{log\_area\_ratio}]^\top \in \mathbb{R}^8$$

### 4.2 Spatial MLP Architecture
- In `SpatialMLP` (`vynix_fewshot_adapter.py:249–266`):
  Maps the 8D relative spatial vector to 600 spatial interaction probability priors:
  $$\text{net}(g) = \text{Sigmoid}\Big(\text{Linear}_{64 \to 600}\big(\text{ReLU}\big(\text{LayerNorm}_{64}\big(\text{Linear}_{8 \to 64}(g)\big)\big)\big)\Big) \in [0, 1]^{600}$$
  Parameters verified from checkpoint:
  - `net.0.weight` $\in \mathbb{R}^{64 \times 8}$, `net.0.bias` $\in \mathbb{R}^{64}$
  - `net.1.weight` $\in \mathbb{R}^{64}$, `net.1.bias` $\in \mathbb{R}^{64}$ (LayerNorm)
  - `net.3.weight` $\in \mathbb{R}^{600 \times 64}$, `net.3.bias` $\in \mathbb{R}^{600}$

---

## 5. Non-Parametric Memory Cache & Logit Blending

### 5.1 Cache Structure
- In `Vynix3StreamAdapter` (`vynix_fewshot_adapter.py:515–533`):
  - Cache Keys: $K_{\text{cache}} \in \mathbb{R}^{N_{\text{cache}} \times 1536}$ (fused 1536-D visual embeddings).
  - Cache Values (one-hot HOI labels): $V_{\text{cache}} \in \mathbb{R}^{N_{\text{cache}} \times 600}$.
  - $N_{\text{cache}} = 19,824$ (Vynix 1.0) and $20,006$ (Vynix 2.0).

### 5.2 Cache Similarity & Affinity
- In `Vynix3StreamAdapter.forward()` (`vynix_fewshot_adapter.py:550–553`):
  Cosine affinity between query $f_{\text{fused}} \in \mathbb{R}^{1536}$ and cached keys:
  $$A = \exp\left(-\beta \cdot (1 - f_{\text{fused}} \cdot K_{\text{cache}}^\top)\right) \in \mathbb{R}^{1 \times N_{\text{cache}}}$$
  where $\beta$ is a learned sharpness parameter ($\beta \approx 5.56$ for v1.0, $5.59$ for v2.0).
- Cache retrieval logits:
  $$S_{\text{cache}} = A \cdot V_{\text{cache}} \in \mathbb{R}^{1 \times 600}$$

### 5.3 Blended Logits Formulation
- In `Vynix3StreamAdapter.forward()` (`vynix_fewshot_adapter.py:555–563`):
  $$S_{\text{blended}} = S_{\text{vlm}} + \alpha \cdot S_{\text{cache}}$$
  When spatial MLP is enabled:
  $$p_{\text{spatial}} = \text{SpatialMLP}(g) \in [0, 1]^{600}$$
  $$S_{\text{final}} = S_{\text{blended}} + 0.5 \cdot \ln(p_{\text{spatial}} + 10^{-6})$$
  where $\alpha$ is a learned weighting parameter ($\alpha \approx 0.955$ for v1.0, $0.939$ for v2.0).
- Softmax Normalization (`vynix_fewshot_adapter.py:822, evaluate_test_unified.py:330`):
  $$P_{\text{base}} = \text{softmax}(S_{\text{final}}) \in \mathbb{R}^{600}$$

---

## 6. Gating Mechanisms & Score Calibration

### 6.1 Soft Geometric Veto Gate ($\gamma_{\text{geo}}$)
- In `compute_soft_geometric_gate()` (`vynix_fewshot_adapter.py:168–189`):
  - Normalized edge-to-edge distance:
    $$\text{dist}_{\text{edge}}(b_h, b_o) = \sqrt{\max(0, x_{o1} - x_{h2}, x_{h1} - x_{o2})^2 + \max(0, y_{o1} - y_{h2}, y_{h1} - y_{o2})^2}$$
    $$d_{\text{box}} = \frac{\text{dist}_{\text{edge}}(b_h, b_o)}{\sqrt{W^2 + H^2}}$$
  - Gate logic:
    $$\gamma_{\text{geo}}(v, \text{IoU}, d_{\text{box}}) = \begin{cases} 
      1.0 & \text{if } v \in \mathcal{V}_{\text{non-contact}} \text{ or } \text{IoU} > 0 \\
      \exp\left(-\left(\frac{d_{\text{box}}}{\sigma}\right)^2\right) & \text{if } v \in \mathcal{V}_{\text{contact}} \text{ and } \text{IoU} = 0
    \end{cases}$$
  *Formula Discrepancy Note*: The source code computes $\exp(-(d_{\text{box}}/\sigma)^2)$ (`vynix_fewshot_adapter.py:187`), which is mathematically equivalent to the paper formula $\exp(-d_{\text{box}}^2 / (2\sigma_{\text{paper}}^2))$ with $\sigma = \sigma_{\text{paper}} \sqrt{2}$.

### 6.2 Object Semantic Identity Gate ($G_{\text{obj}}$)
- In `evaluate_test_unified.py:348–356` (Vynix 2.0 only):
  - Text embedding of target object name: $t_{\text{obj}} \in \mathbb{R}^{512}$ (`a photo of a {obj}`).
  - Normalized object visual feature: $f_o \in \mathbb{R}^{512}$.
  - Cosine similarity:
    $$\text{sim}_{\text{obj}} = \max(0, \, f_o \cdot t_{\text{obj}})$$
  - Gate value:
    $$G_{\text{obj}} = (\text{sim}_{\text{obj}})^{\gamma_{\text{obj}}}$$
    where $\gamma_{\text{obj}} = 0.10$ for Vynix 2.0; $\gamma_{\text{obj}} = 0.00 \implies G_{\text{obj}} = 1.0$ for Vynix 1.0.

### 6.3 Detector Calibration ($\lambda$)
- Power calibration on joint detection scores (`evaluate_test_unified.py:345`):
  $$S_{\text{det}} = (s_h \cdot s_o)^\lambda$$
  where $\lambda = 1.00$ for Vynix 1.0; $\lambda = 0.90$ for Vynix 2.0.

---

## 7. Candidate Expansion, Final HOI Scoring, and Ranking

### 7.1 How Many HOI Classes Are Emitted Per Pair?
- In `evaluate_adapter()` (`vynix_fewshot_adapter.py:808–810`) and `evaluate_test_unified.py:263–266`:
  - Each detected object has an identified COCO class $c_{\text{coco}}$, mapped to HICO object $c_{\text{hico}}$.
  - The pair is expanded **only** across the valid HOI categories for that object:
    $$\mathcal{H}(c_{\text{hico}}) = \text{meta.obj\_to\_entries}[c_{\text{hico}}]$$
  - In HICO-DET, each object category has between 1 and 24 valid verbs (mean $\approx 7.5$ verbs per object; e.g., `bicycle` has 24 valid HOI categories, `toaster` has 2).
  - **Exact Emission**: Exactly $|\mathcal{H}(c_{\text{hico}})|$ HOI classes are emitted for that pair. All other $600 - |\mathcal{H}(c_{\text{hico}})|$ classes have an implicit score of 0 and are omitted from candidate ranking.

### 7.2 Final HOI Score Formula
For each candidate triplet $(b_h, b_o, c)$ where $c = (v_c, o_c) \in \mathcal{H}(c_{\text{hico}})$:
$$S_{\text{HOI}}(b_h, b_o, c) = (s_h \cdot s_o)^\lambda \cdot P_{\text{base}}[c] \cdot \gamma_{\text{geo}}(v_c, \text{IoU}, d_{\text{box}}) \cdot G_{\text{obj}}$$
Triplets emitted to the evaluator:
$$(b_h, b_o, c, S_{\text{HOI}})$$

### 7.3 Ranking and Average Precision Evaluation Rules
- For each image, up to $50 \times \bar{V} \approx 375$ triplets are emitted.
- All predictions across all $9,658$ test images are accumulated.
- For each HOI class $c \in \{0, \dots, 599\}$:
  1. All predictions across the dataset with class $c$ are sorted descending by score.
  2. Predictions are matched sequentially to unassigned ground-truth instances of class $c$ in the corresponding image.
  3. A prediction is a **True Positive** iff $\min(\text{IoU}(b_h, b_{h,\text{gt}}), \, \text{IoU}(b_o, b_{o,\text{gt}})) \ge 0.5$; otherwise **False Positive**.
- **AP Evaluation Rules**:
  - **Rule A (VOC 2010+ / COCO continuous all-points interpolation)** (`review_response/hico_evaluator.py:42–63`):
    $$AP_{\text{cont}} = \sum_{k} (r_{k+1} - r_k) \max_{\tilde{r} \ge r_{k+1}} p(\tilde{r})$$
  - **Rule B (VOC 2007 11-point interpolation `11P`, standard in `fredzzhang/pocket` and `fredzzhang/upt:161`)**:
    $$AP_{11\text{P}} = \frac{1}{11} \sum_{t \in \{0, 0.1, \dots, 1.0\}} \max_{\tilde{r} \ge t} p(\tilde{r})$$
  Both rules yield consistent rankings; on the full test set, $AP_{11\text{P}}$ produces slightly higher overall mAP (+0.56% on v1.0, +0.53% on v2.0) due to 11-point sampling discretization.
