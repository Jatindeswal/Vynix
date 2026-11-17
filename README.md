# Project Vynix 👁️🧠

> [!IMPORTANT]
> **Vynix v3 Release & Reproducibility Audit Notice: Standard Box-Level Default Protocol**
> - Earlier project drafts reported **image-level multi-label retrieval AP** ($30.31\%$ for v1.0, $34.80\%$ for v2.0) using a non-standard pseudo-rare split ($150$ classes with $h \bmod 4 == 0$). These image-level metrics evaluate whole-image HOI presence without bounding box spatial localization and are **not directly comparable to published box-level HICO-DET benchmark tables**.
> - In **Vynix v3**, evaluation conforms strictly to the official **HICO-DET Standard Box-Level Default mAP Protocol** ($\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$ on all $9,658$ test images, evaluated across the standard 600 Full / 138 Rare / 462 Non-Rare classes):
>   - **Vynix 1.0 (YOLOv8m + CLIP ViT-B/16):** **12.23%** (VOC10+ Continuous mAP) / **12.80%** (VOC07 11-point interpolated mAP `11P`).
>   - **Vynix 2.0 (YOLOv8x + CLIP ViT-B/16):** **14.76%** (VOC10+ Continuous mAP) / **15.29%** (VOC07 11-point interpolated mAP `11P`).
> - Complete verified metrics, full ablation suites, and hyperparameter provenance are documented in [review_response/FINAL_TABLES.md](review_response/FINAL_TABLES.md), [PAPER_TO_CODE.md](PAPER_TO_CODE.md), and [results/MANIFEST.csv](results/MANIFEST.csv).

> **Few-Shot Human-Object Interaction (HOI) Detection with Geometric Hallucination Veto**

[![Release](https://img.shields.io/badge/Release-v3.0-blue.svg)](https://github.com/Jatindeswal/Vynix/releases)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open Interactive Presentation & Benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Vynix_Interactive_Presentation_and_Benchmark.ipynb)
[![Interactive Slides (HTML5)](https://img.shields.io/badge/Presentation-Interactive%20HTML5%20Deck-cyan?logo=html5)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_Presentation_Review.html)
[![Interactive Architecture Diagram](https://img.shields.io/badge/Architecture-Interactive%20Diagram%20(Archify)-purple?logo=diagramsdotnet)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_System_Architecture.html)

**Project Vynix** is an open-source Human-Object Interaction (HOI) detection system engineered to suppress Vision-Language Model (VLM) spatial hallucinations using continuous geometric reasoning, multi-stream visual adaptation, and non-parametric exemplar caching.

---

## 🏛️ System Architecture

Project Vynix implements a 5-stage decoupled reasoning pipeline:

```
                                  ┌────────────────────────┐
                                  │    Input RGB Image     │
                                  └───────────┬────────────┘
                                              │
                               [Stage 1: Proposal Detection]
                                    YOLOv8 (COCO 80 classes)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
             [Human Bounding Box]                          [Object Bounding Box]
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              │
                             [Stage 2: 3-Stream Visual Fusion]
                        Human Crop (512-d) || Object Crop (512-d) || Union Crop (512-d)
                                 CLIP ViT-B/16 (1536-d fused)
                                              │
                             [Stage 3: Spatial Geometry MLP]
                        8D Geometry Vector -> Interaction Prior P_spatial (600-d)
                                              │
                             [Stage 4: Residual Exemplar Cache]
                        Non-Parametric Affinity Blending + Learned (alpha, beta)
                                              │
                             [Stage 5: Geometric Veto Gate]
                        Continuous Gaussian Attenuation for Zero-IoU Contact Verbs
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Final Grounded Verdict │
                                  └────────────────────────┘
```

### Mathematical Formulation:

1. **Stage 1 — Candidate Proposal Extraction:**
   YOLOv8 extracts human candidates $b_h = (x_1, y_1, x_2, y_2)$ with confidence $s_h \ge 0.25$ and object candidates $b_o$ with confidence $s_o \ge 0.25$ belonging to COCO categories $c_o \in \{0, \dots, 79\}$.

2. **Stage 2 — 3-Stream Multi-Crop Visual Encoding:**
   Each candidate pair yields three crops: human $I_h$, object $I_o$, and contextual union $I_u$. A frozen CLIP ViT-B/16 visual encoder maps them to unit-norm embeddings $f_h, f_o, f_u \in \mathbb{R}^{512}$. They are concatenated and normalized:
   $$f_{\text{vis}} = \frac{[f_h \parallel f_o \parallel f_u]}{\sqrt{3}} \in \mathbb{R}^{1536}$$

3. **Stage 3 — Continuous Spatial Geometry MLP:**
   An 8-D normalized spatial relationship vector is extracted:
   $$g = \left[\frac{c_{ox}-c_{hx}}{W}, \frac{c_{oy}-c_{hy}}{H}, \frac{w_h}{W}, \frac{h_h}{H}, \frac{w_o}{W}, \frac{h_o}{H}, \text{IoU}(b_h, b_o), \ln\left(\frac{\max(w_h h_h, 1)}{\max(w_o h_o, 1)}\right)\right] \in \mathbb{R}^8$$
   A 2-layer MLP ($\text{Linear}(8, 64) \to \text{LayerNorm}(64) \to \text{ReLU} \to \text{Linear}(64, 600) \to \text{Sigmoid}$) maps $g$ to spatial prior likelihoods $p_{\text{spatial}} \in [0, 1]^{600}$.

4. **Stage 4 — Residual Exemplar Cache & Adaptive Blending:**
   Non-parametric exemplar keys $K_{\text{cache}} \in \mathbb{R}^{N \times 1536}$ and one-hot multi-label targets $V_{\text{cache}} \in \{0, 1\}^{N \times 600}$ provide few-shot memory. The blended logits are computed as:
   $$\text{logits} = S_{\text{vlm}} + \alpha \cdot \exp\left(-\beta (1 - f_{\text{vis}} K_{\text{cache}}^T)\right) V_{\text{cache}} + 0.5 \ln(p_{\text{spatial}} + 10^{-6})$$
   where $S_{\text{vlm}} = (f_u / \sqrt{3}) W_{\text{text}}^T \in \mathbb{R}^{600}$, and $\alpha, \beta$ are learned scalar parameters ($\alpha \approx 0.94 - 0.95$, $\beta \approx 5.56 - 5.59$).

5. **Stage 5 — Soft Continuous Geometric Veto Gate:**
   Physical contact interactions ($\mathcal{V}_{\text{contact}}$, 59 verbs) require physical proximity. Hypotheses with $\text{IoU}(b_h, b_o) = 0$ are attenuated via a smooth Gaussian edge-distance decay:
   $$\gamma(v, \text{IoU}, d_{\text{box}}) = \begin{cases} 
   1.0 & \text{if } v \notin \mathcal{V}_{\text{contact}} \lor \text{IoU} > 0.0 \\ 
   \exp\left(-\left(\frac{d_{\text{box}}}{\sigma}\right)^2\right) & \text{if } v \in \mathcal{V}_{\text{contact}} \land \text{IoU} = 0.0 
   \end{cases}$$
   Final HOI score:
   $$S_{\text{HOI}}(h, o, a) = M_{c_o, a} \cdot (s_h s_o)^\lambda \cdot G_{\text{obj}}(o) \cdot \gamma(a, \text{IoU}, d_{\text{box}}) \cdot \text{softmax}(\text{logits})_a$$

---

## 📊 Benchmark Results (Official HICO-DET Test Set)

Evaluated across all **9,658 test images** under the official **HICO-DET Box-Level Default setting** ($\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$):

### Primary Benchmark Table
| Model Release | Detector | Vision Backbone | Full mAP (Continuous) | Rare mAP (138) | Non-Rare mAP (462) | Full mAP (11-Point 11P) | Rare (11P) | Non-Rare (11P) | Hallucinations Vetoed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vynix 1.0** | YOLOv8m | CLIP ViT-B/16 | **12.23%** | 11.14% | 12.56% | **12.80%** | 10.69% | 13.42% | 265,189 |
| **Vynix 2.0** | **YOLOv8x** | **CLIP ViT-B/16** | **14.76%** | **15.37%** | **14.58%** | **15.29%** | **14.71%** | **15.46%** | **272,946** |
| *YOLOv8x + v1 Cache* | YOLOv8x | CLIP ViT-B/16 | 13.25% | 13.46% | 13.19% | 13.70% | 12.59% | 14.03% | 272,946 |

### Ablation Suite (Vynix 2.0 on Full 9,658 Test Images)
| Ablation Setting | Full mAP (Cont.) | Rare (138) | Non-Rare (462) | Full mAP (11P) | Impact / Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Vynix 2.0 (Proposed)** | **14.76%** | **15.37%** | **14.58%** | **15.29%** | Complete calibrated pipeline |
| Without Geometric Gate ($\sigma = \infty$) | 14.12% | 14.97% | 13.87% | 14.65% | **-0.64% drop** (hallucinations unchecked) |
| Hard Binary Gate ($\text{IoU} > 0$ strictly) | 14.49% | 15.34% | 14.23% | 15.01% | **-0.27% drop** (kills adjacent contact interactions) |
| Without Object Gate ($G_{\text{obj}} = 1$) | 14.73% | 15.47% | 14.51% | 15.25% | -0.03% drop |
| Detector Score Only ($\lambda = 1.0$) | 14.75% | 15.37% | 14.57% | 15.28% | -0.01% drop |
| Raw Baseline (No gates / calibration) | 14.05% | 15.01% | 13.77% | 14.57% | **-0.71% drop** |

---

## ⚡ Execution Latency & Profiling

Profiled cleanly on an isolated **NVIDIA RTX 4070 Ti** GPU over 300 test images (averaged over two independent benchmark passes):

| Model Configuration | Total Latency | Effective FPS | Detector Stage | CLIP Encoding Stage | Adapter & Veto Stage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vynix 1.0 (YOLOv8m)** | **21.19 ms/img** | **~47.2 FPS** | 7.91 ms | 12.39 ms | 0.74 ms |
| **Vynix 2.0 (YOLOv8x)** | **30.83 ms/img** | **~32.4 FPS** | 17.50 ms | 12.38 ms | 0.76 ms |

---

## 📁 Repository Structure (Vynix v3)

```
Vynix/
├── src/                               # Core modular package
│   ├── data/
│   │   ├── class_splits.py            # Canonical 138 Rare / 462 Non-Rare splits, 80 COCO, 117 verbs
│   │   └── predicates.py              # Canonical 59 contact / 58 non-contact verb partition
│   ├── models/
│   │   ├── vynix_fewshot_adapter.py   # Multi-stream visual fusion and Tip-Adapter-F core
│   │   └── checkpoint_loader.py       # Unified loader with strict [N, 1536] shape validation
│   ├── evaluation/
│   │   └── hico_evaluator.py          # Official Box-Level Default setting evaluator (VOC10+ & 11P)
│   └── config/
│       ├── paths.py                   # Environment variable path resolver (no hardcoded drives)
│       └── hyperparameters.py         # Loader for frozen hyperparameters
├── configs/
│   ├── paths.yaml                     # Centralized paths specification
│   └── frozen_hyperparameters.json    # Frozen inference parameters (val_10 tuned)
├── tests/                             # Automated reproducibility test suite
│   ├── test_shapes.py                 # Tensor shape assertions across all 4 checkpoints
│   ├── test_predicates.py             # Predicate count (117) and contact gate routing tests
│   ├── test_evaluator.py              # Analytic toy case (5/6 PR-AUC), IoU veto, duplicate suppression
│   ├── test_legacy_conf.py            # Ultralytics default confidence behavior verification
│   └── test_regression.py            # 50-image bit-exact regression verification
├── results/
│   └── MANIFEST.csv                   # Command, seed, and file manifest for every reported metric
├── review_response/                   # Audit tables, disclosures, and valuation scripts
│   ├── FINAL_TABLES.md                # Official benchmark tables and ablation results
│   ├── DISCLOSURES.md                 # Full disclosures on splits, gates, and thresholds
│   └── frozen_hyperparameters.json    # Canonical frozen hyperparameters file
├── archive/                           # Preserved legacy and scratch scripts (via git mv)
│   └── INDEX.md                       # Comprehensive archive catalog and rationales
├── regression_before.pt               # Deterministic 50-image regression baseline
├── PAPER_TO_CODE.md                   # Systematic 13-point audit matrix linking paper to code
├── CLEANUP_REPORT.md                  # Comprehensive Phase 2 refactoring report
├── STALE_CLAIMS.md                    # Catalog of historical claims and superseded numbers
├── requirements.lock                  # Frozen pip dependency lockfile
└── README.md                          # Project documentation
```

---

## 🧪 Reproducibility & Testing

All unit tests and reproducibility harnesses are executed through standard Python `unittest`:

```bash
# 1. Run all core unit tests (<10 seconds)
python -m unittest tests/test_predicates.py tests/test_evaluator.py tests/test_shapes.py tests/test_legacy_conf.py

# 2. Run the full test suite including the 50-image regression test
python -m unittest discover tests

# 3. Run standalone regression check against baseline
python run_regression.py --compare regression_before.pt
```

---

## 💻 Quickstart & Inference

### Installation
```bash
git clone https://github.com/Jatindeswal/Vynix.git
cd Vynix
pip install -r requirements.txt
```

### Python API
```python
from src.models.checkpoint_loader import load_vynix_checkpoint
from src.config.hyperparameters import get_inference_hyperparameters
from src.data.predicates import is_contact_verb

# Load Vynix 2.0 checkpoint with shape validation
model = load_vynix_checkpoint("v2.0", device="cuda")
print(f"Loaded {model['num_entries']} cached exemplars. Alpha: {model['alpha']:.4f}")

# Retrieve frozen inference parameters
params = get_inference_hyperparameters("vynix_2.0")
print(f"Inference config: sigma={params['sigma']}, lambda={params['lambda']}")
```

---

## 📜 Provenance & Audit Artifacts

- [review_response/FINAL_TABLES.md](review_response/FINAL_TABLES.md): Full test set evaluation tables, ablations, and veto breakdown.
- [PAPER_TO_CODE.md](PAPER_TO_CODE.md): Paper claim-by-claim verification against code implementations.
- [results/MANIFEST.csv](results/MANIFEST.csv): Exact commands to regenerate every number reported.
- [CLEANUP_REPORT.md](CLEANUP_REPORT.md): Progression log of the Phase 2 declutter and consistency hardening.
- [review_response/DISCLOSURES.md](review_response/DISCLOSURES.md): Transparent disclosures addressing all peer-review remarks.

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
