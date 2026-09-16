# Project Vynix 👁️🧠

> **Few-Shot Human-Object Interaction (HOI) Detection with Geometric Hallucination Veto**

[![Open Interactive Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Vynix_Colab_Demo.ipynb)
[![Open Error Analysis & Few-Shot in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Vynix_Error_Analysis_and_FewShot.ipynb)
[![Open Dataset Benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jatindeswal/Vynix/blob/main/Vynix_HICO_DET_Benchmark.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Project Vynix** is a lightweight, zero-shot/few-shot Human-Object Interaction (HOI) detection pipeline designed to eliminate vision-language model (VLM) spatial hallucinations using pure geometric reasoning.

---

## 🏛️ Pipeline Architecture

```
                                      ┌────────────────────────┐
                                      │   Input RGB Image      │
                                      └───────────┬────────────┘
                                                  │
                                   [Stage 1: Node Extraction]
                                          YOLOv8-nano
                                                  │
                           ┌──────────────────────┴──────────────────────┐
                           ▼                                             ▼
                 [Human Bounding Box]                          [Object Bounding Box]
                           │                                             │
                           └──────────────────────┬──────────────────────┘
                                                  │
                                    [Stage 2: Spatial Graph]
                                      Centroids, Dist, IoU
                                                  │
                                   [Stage 3: VLM Alignment]
                                   CLIP (ViT-B/32) on Union
                                                  │
                                  [Stage 4: Vynix Logic Gate]
                             Geometric Veto if IoU == 0 on Contact
                                                  │
                                                  ▼
                                      ┌────────────────────────┐
                                      │ Final Grounded Verdict │
                                      └────────────────────────┘
```

1. **Stage 1 — Node Extraction ("The Eyes")**: Uses YOLOv8-nano to detect all human agents ($c=0$) and object patients ($c \in [1, 79]$).
2. **Stage 2 — Scene Graph Math ("The Brain")**: Computes bounding-box centroids, Euclidean distances, and Intersection-over-Union ($\text{IoU}$) for candidate interaction pairs.
3. **Stage 3 — VLM Alignment ("The Translator")**: Crops the minimum enclosing union bounding box and scores it against dynamic verb-object prompts using OpenAI CLIP (`openai/clip-vit-base-patch32`).
4. **Stage 4 — The Vynix Logic Gate ("The Arbiter")**: Applies a hard geometric override. If CLIP predicts an active physical contact interaction (e.g. *holding*, *carrying*, *riding*) but the spatial overlap $\text{IoU} = 0.0$, the gate vetoes the hallucinated interaction:

$$
\mathcal{V}(v, \text{IoU}) = \begin{cases} 
\text{"No Interaction"} & \text{if } v \in \mathcal{C}_{\text{contact}} \;\land\; \text{IoU} = 0.0 \\ 
\operatorname{argmax}(\text{CLIP}) & \text{otherwise} 
\end{cases}
$$

---

## 📊 HICO-DET Full Benchmark Performance

Evaluated against the complete **HICO-DET** test set (**9,658 images**, **600 HOI categories**):

| Split | Classes | Mean Average Precision (mAP) |
| :--- | :---: | :---: |
| **Full** | **600** | **22.17%** |
| **Rare** ($< 10$ train samples) | **155** | **18.57%** |
| **Non-Rare** ($\ge 10$ train samples) | **445** | **23.42%** |

### 🛡️ Hallucinations Prevented: **319,803**
- **Top Overridden Verbs**: `hold` (44,979), `carry` (31,730), `wash` (27,652), `ride` (22,726), `sit_on` (17,985)
- **Top Overridden Objects**: `car` (40,524), `bicycle` (30,680), `chair` (23,920), `cup` (22,687), `horse` (15,680)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Jatindeswal/Vynix.git
cd Vynix
pip install -r requirements.txt
```

### 2. Run the 2-Image Demo Prototype
```bash
python vynix_prototype.py
```
*Generates annotated visual proof images: `vynix_proof_holding.jpg` and `vynix_proof_not_holding.jpg`.*

### 3. Run the Full HICO-DET Evaluation
```bash
# Quick smoke test (10 images)
python evaluate_vynix_full.py --limit-samples 10 --device cpu

# Full 9,658-image evaluation with GPU acceleration
python evaluate_vynix_full.py --device cuda --batch-size 32 --output-csv vynix_hico_results.csv
```

---

## 📁 Repository Structure

```
Vynix/
├── vynix_prototype.py        # Clean 170-line demo & viva prototype
├── evaluate_vynix_full.py    # Full 600-class HICO-DET benchmark evaluator
├── vynix_hico_results.csv    # Official benchmark AP per-class results
├── test_image.jpg            # Benchmark sample (Holding)
├── test_image1.jpg           # Benchmark sample (Not Holding)
├── vynix_proof_holding.jpg   # Annotated proof output
├── vynix_proof_not_holding.jpg # Annotated proof output
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 📜 License
This project is licensed under the MIT License.
