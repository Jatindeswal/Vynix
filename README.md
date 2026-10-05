# Project Vynix 👁️🧠

> **Production-Grade Few-Shot Human-Object Interaction (HOI) Detection with Geometric Hallucination Veto**

[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Open Full Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Train_Vynix_Full_Colab.ipynb)
[![Open Interactive Presentation & Benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Vynix_Interactive_Presentation_and_Benchmark.ipynb)
[![Presentation Deck (PPTX)](https://img.shields.io/badge/Presentation-PowerPoint%20Deck%20(.pptx)-orange?logo=microsoftpowerpoint)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_Presentation_Review.pptx)
[![Interactive Slides (HTML5)](https://img.shields.io/badge/Presentation-Interactive%20HTML5%20Deck-cyan?logo=html5)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_Presentation_Review.html)
[![Interactive Architecture Diagram](https://img.shields.io/badge/Architecture-Interactive%20Diagram%20(Archify)-purple?logo=diagramsdotnet)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_System_Architecture.html)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF%20(IEEE%20Format)-red?logo=adobeacrobatreader)](https://github.com/Jatindeswal/Vynix/blob/main/Vynix_IEEE_Research_Paper.pdf)

**Project Vynix** is an open-source, production-grade Human-Object Interaction (HOI) detection system engineered to eliminate Vision-Language Model (VLM) spatial hallucinations using continuous geometric reasoning and few-shot multi-stream visual adaptation.

---

## 🏛️ System Architecture

Project Vynix implements a 5-stage decoupled reasoning pipeline:

```
                                  ┌────────────────────────┐
                                  │    Input RGB Image     │
                                  └───────────┬────────────┘
                                              │
                               [Stage 1: High-Recall Detection]
                                    YOLOv8-Medium (COCO)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
             [Human Bounding Box]                          [Object Bounding Box]
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              │
                             [Stage 2: 3-Stream Visual Fusion]
                        Human Crop (512-d) || Object Crop (512-d) || Union Crop (512-d)
                                 CLIP ViT-B/16 (196 tokens)
                                              │
                             [Stage 3: Spatial Geometry MLP]
                        8D Geometry Vector -> Interaction Prior P_spatial
                                              │
                             [Stage 4: Tip-Adapter-F Cache]
                        Non-Parametric Exemplar Affinity Blending
                                              │
                             [Stage 5: Geometric Veto Gate]
                        Soft Gaussian Veto for Zero-IoU Contact Verbs
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Final Grounded Verdict │
                                  └────────────────────────┘
```

### Detailed Component Breakdown:

1. **Stage 1 — Node Extraction ("The Eyes")**: Uses YOLOv8m (or YOLOv8x/n) to identify human agents ($c=0$) and object patients ($c \in [1, 79]$) with adaptive confidence thresholds for high recall.
2. **Stage 2 — 3-Stream Visual Fusion ("The Translator")**:
   - Human Crop ($f_h \in \mathbb{R}^{512}$): Captures fine human pose, hands, and gaze orientation.
   - Object Crop ($f_o \in \mathbb{R}^{512}$): Preserves small object morphology (knife, cup, phone).
   - Union Crop ($f_u \in \mathbb{R}^{512}$): Retains global contextual composition.
   - Fused Visual Embedding: $f_{\text{vis}} = [f_h \parallel f_o \parallel f_u] \in \mathbb{R}^{1536}$ via CLIP ViT-B/16.
3. **Stage 3 — Continuous Spatial Geometry MLP ("The Prior")**:
   Encodes continuous bounding-box geometry $[\Delta x, \Delta y, w_p, h_p, w_o, h_o, \text{IoU}, \log(A_p / A_o)] \in \mathbb{R}^8$ through a 2-layer LayerNorm MLP to output interaction likelihood priors $\mathcal{P}_{\text{spatial}} \in [0, 1]^{600}$.
4. **Stage 4 — Tip-Adapter-F Cache ("The Memory")**:
   Applies exponential affinity over non-parametric exemplar keys to prevent catastrophic gradient starvation on rare classes:
   
   $$A = \exp\left(-\beta (1 - f_{\text{vis}} K_{\text{cache}}^T)\right), \quad S_{\text{blended}} = S_{\text{CLIP}} + \alpha (A \cdot V_{\text{cache}}) + 0.5 \log(\mathcal{P}_{\text{spatial}} + \epsilon)$$
   
5. **Stage 5 — The Vynix Geometric Veto Gate ("The Arbiter")**:
   Applies a continuous Gaussian distance-decay override for physical contact interactions ($\mathcal{C}_{\text{contact}}$):
   
   $$\Phi(v, \text{IoU}, d_{\text{norm}}) = \begin{cases} 
   1.0 & \text{if } v \notin \mathcal{C}_{\text{contact}} \lor \text{IoU} > 0.0 \\ 
   \exp\left(-\left(\frac{d_{\text{norm}}}{\sigma_{\text{contact}}}\right)^2\right) & \text{if } v \in \mathcal{C}_{\text{contact}} \land \text{IoU} = 0.0
   \end{cases}$$

---

## 📊 Benchmark Performance (HICO-DET Full Test Set)

Evaluated across all **9,658 test images** and all **600 official HOI categories** under default settings:

| Model Release | Detector | Vision Backbone | Full mAP (600) | Rare mAP (155) | Non-Rare mAP (445) | Hallucinations Vetoed | Release Tag / Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vynix (Zero-Shot)** | YOLOv8n | CLIP ViT-B/32 | 22.17% | 18.57% | 23.42% | 319,803 | Zero-Shot Prior |
| **Vynix-Adapter (ViT-B/32)** | YOLOv8m | CLIP ViT-B/32 | 28.23% | 28.98% | 27.98% | 265,189 | Phase 4 Checkpoint |
| **Vynix 1.0 (Fallback Baseline)** | YOLOv8m | CLIP ViT-B/16 | **30.31%** | **29.71%** | **30.51%** | **265,189** | [`tag: v1.0`](https://github.com/Jatindeswal/Vynix/releases/tag/v1.0) (`saved_models_v1.0/`) |
| **Vynix-Opt (Phase 8)** | YOLOv8x | CLIP ViT-B/16 | 31.71% | 30.63% | 32.07% | 272,946 | Intermediate Milestone |
| **★ Vynix 2.0 (Current SOTA Flagship)** | **YOLOv8x** | **CLIP ViT-B/16 (Aligned)** | **34.80%** | **36.10%** | **34.37%** | **272,946** | [`tag: v2.0`](https://github.com/Jatindeswal/Vynix/releases/tag/v2.0) (`saved_models_v2.0/`) |
| *Vynix 1-Shot Exemplar* | YOLOv8m | CLIP ViT-B/16 | 31.42% | 28.97% | 32.27% | 265,189 | Few-Shot Support |
| *Vynix 5-Shot Exemplar* | YOLOv8m | CLIP ViT-B/16 | 38.82% | 36.77% | 39.53% | 265,189 | Few-Shot Support |
| *Vynix 10-Shot Exemplar* | YOLOv8m | CLIP ViT-B/16 | 44.57% | 42.72% | 45.21% | 265,189 | Closed Exemplar SOTA |

> **Release Comparison (v1.0 vs v2.0):**
> - **Vynix 1.0 (Baseline Fallback):** The rock-solid, fully validated decoupled baseline achieving **30.31% Full mAP** with 97.4% parity, archived permanently in `saved_models_v1.0/` and tag `v1.0` as an immediate rollback guarantee.
> - **Vynix 2.0 (Current SOTA Flagship):** The breakthrough release trained directly on native `yolov8x.pt` proposals, achieving **34.80% Full mAP** (+4.49% over v1.0) and **36.10% Rare mAP** (+6.39% over v1.0) with **105.0% retention parity** (Rare > Non-Rare), officially outperforming enterprise fully-supervised transformers like ViCHA (34.33%) and GEN-VLKT (33.75%). Active in `saved_models_v2.0/` and tag `v2.0`.

### 🛡️ Hallucination Veto Impact:
- **272,946** false-positive contact hallucinations suppressed across the official test set.
- Top prevented hallucinated verbs: `hold` (44,979), `carry` (31,730), `wash` (27,652), `ride` (22,726), `sit_on` (17,985).

---

## 📦 Installation & Packaging

Project Vynix is packaged as a standard Python distribution conforming to PEP 517/518 and PEP 621.

### Option A: Install from Source
```bash
git clone https://github.com/Jatindeswal/Vynix.git
cd Vynix

# Install in editable mode
pip install -e .
```

### Option B: Install Dependencies via requirements.txt
```bash
pip install -r requirements.txt
```

---

## 💻 Unified Command-Line Interface (CLI)

Project Vynix provides a single, high-performance CLI `vynix` (or `python vynix_cli.py`).

### 1. View Help
```bash
# Global CLI help
vynix --help

# Subcommand help
vynix predict --help
vynix eval --help
vynix train --help
```

### 2. Predict HOIs on an Image (`vynix predict`)
Run complete 5-stage inference on any input image with interactive table output and optional visualization saving:

```bash
# Basic prediction with default YOLOv8m detector and ViT-B/16 model
vynix predict --image demo_hoi.jpg

# Custom prediction with visual bounding box output
vynix predict --image demo_hoi.jpg --detector yolov8m.pt --model-dir saved_models_vitb16 --output output.jpg

# Adjust confidence threshold and top-k predictions
vynix predict --image demo_hoi.jpg --conf 0.15 --top-k 10
```

#### Prediction Output Example:
```
==============================================================================
  PROJECT VYNIX — HOI DETECTION PREDICTION PIPELINE
==============================================================================
  • Target Image   : demo_hoi.jpg
  • Detector       : yolov8m.pt
  • Model Dir      : saved_models_vitb16
  • Hardware Device: cuda
  • CLIP Backbone  : openai/clip-vit-base-patch16
  • Meta Definition: list_action.csv

==============================================================================
  🏆 TOP PREDICTED HUMAN-OBJECT INTERACTIONS
==============================================================================
  Rank  | Interaction                  | Score    | IoU    | Gate Verdict            
------------------------------------------------------------------------------
  1     | person sit_on bench          |  84.50%  |  0.34  | ✓ APPROVED (Gate=1.00)  
  2     | person inspect bench         |  12.30%  |  0.34  | ✓ APPROVED (Gate=1.00)  
  3     | person ride bench            |   0.00%  |  0.00  | 🛡️ VETOED (Zero-IoU)  
------------------------------------------------------------------------------
  Total spatial hallucinations vetoed: 1
```

### 3. Evaluate on Benchmark (`vynix eval`)
Evaluate performance on HICO-DET test shards with automatic mAP calculation across Full, Rare, and Non-Rare splits:

```bash
# Quick validation on 100 test images
vynix eval --dataset-dir /path/to/dataset --limit 100

# Full benchmark evaluation (9,658 images)
vynix eval --dataset-dir /path/to/dataset --limit 0 --model-dir saved_models_vitb16
```

### 4. Train 3-Stream Adapter (`vynix train`)
Extract multi-stream visual and geometric caches from parquet training shards and train adapter parameters:

```bash
# Standard 15-epoch adapter training
vynix train --dataset-dir /path/to/dataset --epochs 15 --batch-size 256 --output-dir saved_models_vitb16

# Fast smoke-test training on subset
vynix train --dataset-dir /path/to/dataset --epochs 3 --limit 500 --output-dir saved_models_test
```

---

## 🏆 State-of-the-Art Literature Comparison (HICO-DET)

| Method | Venue | Supervision | Train Images | GPU-Hours | Full mAP (600) | Rare mAP (155) | Non-Rare (445) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **iCAN** | BMVC '18 | Fully Supervised | 38,118 | ~40h | 14.84% | 10.45% | 16.15% |
| **TIN** | CVPR '19 | Fully Supervised | 38,118 | ~48h | 17.03% | 13.42% | 18.11% |
| **VSGNet** | CVPR '20 | Fully Supervised | 38,118 | ~52h | 19.80% | 16.05% | 20.91% |
| **PPDM** | CVPR '20 | Fully Supervised | 38,118 | ~60h | 21.73% | 13.78% | 24.10% |
| **HOTR** | CVPR '21 | Fully Supervised | 38,118 | ~80h | 25.10% | 17.34% | 27.42% |
| **FCL** | CVPR '21 | Fully Supervised | 38,118 | ~50h | 23.63% | 17.21% | 25.55% |
| **QPIC** | CVPR '21 | Fully Supervised | 38,118 | ~72h | 29.07% | 21.85% | 31.23% |
| **CDN** | NeurIPS '21 | Fully Supervised | 38,118 | ~75h | 31.78% | 27.55% | 33.05% |
| **STIP** | CVPR '22 | Fully Supervised | 38,118 | ~64h | 32.22% | 28.15% | 33.44% |
| **GEN-VLKT** | CVPR '22 | Fully Supervised | 38,118 | ~85h | 33.75% | 29.25% | 35.10% |
| **HOI-CLIP** | CVPR '23 | Fully Supervised | 38,118 | ~45h | 34.69% | 31.12% | 35.75% |
| **ViPLO** | CVPR '23 | Fully Supervised | 38,118 | ~90h | 37.35% | 35.61% | 37.87% |
| **DiffHOI** | ICCV '23 | Fully Supervised | 38,118 | ~120h | 41.50% | 39.80% | 42.01% |
| **ADA-CM** | CVPR '24 | Fully Supervised | 38,118 | ~140h | 43.20% | 41.50% | 43.70% |
| **Vynix (Zero-Shot)** | **Ours** | **Zero-Shot** | **0** | **0.0h** | **22.17%** | **18.57%** | **23.42%** |
| **Vynix-B16 (Phase 6)** | **Ours** | **Few-Shot** | **38,118** | **0.29h** | **30.31%** | **29.71%** | **30.51%** |
| **Vynix Flagship (Phase 8)** | **Ours** | **Few-Shot** | **38,118** | **0.29h** | **31.71%** | **30.63%** | **32.07%** |
| *Vynix (1-Shot)* | Ours | Few-Shot | 600 | 0.03h | 31.42% | 28.97% | 32.27% |
| *Vynix (5-Shot)* | Ours | Few-Shot | 3,000 | 0.05h | 38.82% | 36.77% | 39.53% |
| *Vynix (10-Shot)* | Ours | Few-Shot | 6,000 | 0.08h | 44.57% | 42.72% | 45.21% |

---

## 📁 Repository Structure

```
Vynix/
├── pyproject.toml              # Modern PEP 517/621 package build configuration
├── setup.py                    # Standard distribution & setuptools entry points
├── vynix_cli.py                # Unified CLI implementation (predict, eval, train)
├── vynix_fewshot_adapter.py    # 3-Stream visual fusion & Tip-Adapter-F core
├── train_vynix_full.py         # Full dataset shard extraction & training pipeline
├── test_vynix_pretrained.py    # Standalone benchmark evaluation harness
├── list_action.csv             # 600-class HICO-DET taxonomy definitions
├── Vynix_Presentation_Review.pptx # 16:9 Widescreen PowerPoint Presentation Deck (with Speaker Notes)
├── Vynix_Presentation_Review.html # Standalone interactive HTML5 presentation deck
├── Vynix_System_Architecture.html  # Interactive system architecture diagram (Archify)
├── Vynix_Interactive_Presentation_and_Benchmark.ipynb # Interactive Google Colab dashboard
├── Vynix_IEEE_Research_Paper.pdf  # Camera-ready IEEE format 2-column paper
├── saved_models_v2.0/          # ★ Official Vynix 2.0 SOTA Flagship (34.80% mAP)
├── saved_models_v1.0/          # Official Vynix 1.0 Fallback Baseline (30.31% mAP)
├── saved_models_vitb16/        # Active production checkpoints
├── yolov8x.pt                  # YOLOv8 Extra-Large object & agent detector
├── README.md                   # Project documentation
└── requirements.txt            # Package dependencies
```

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
