# Vynix Reproducibility & Peer-Review Audit: Final Evaluation Tables

**Protocol Specifications**:
- **Dataset**: HICO-DET Test Set ($9,658$ images).
- **Evaluation Settings**:
  - **Standard Box-Level Default mAP**: Prediction is a triplet $(\text{box}_h, \text{box}_o, \text{hoi\_id}, \text{score})$; marked True Positive iff $\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$ against an unmatched ground-truth annotation of the same HOI class. Evaluated across the official benchmark splits: Full ($600$ classes), Standard Rare ($138$ classes, $<10$ training instances), and Standard Non-Rare ($462$ classes, $\ge 10$ training instances).
  - **Legacy Image-Level AP**: Prediction is an image-level $(\text{hoi\_id}, \text{max\_score})$; marked True Positive if $\text{hoi\_id}$ is present in the image annotation. Evaluated on Full ($600$), Standard Rare ($138$), Standard Non-Rare ($462$), and Legacy Pseudo-Rare ($150$ classes, $h \bmod 4 == 0$).
- **Hardware & Environment**:
  - GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM).
  - Software: Python 3.13.12, PyTorch 2.13.0+cu126, Transformers 5.14.1, Ultralytics 8.4.107, NumPy 2.4.6.
  - Frozen Hyperparameters Commit: `bb3228a` / `e6e6db7` (`review_response/frozen_hyperparameters.json`).
  - Execution Status: Single-pass test evaluation completed for Vynix 1.0 (`task-4618`) and Vynix 2.0 (`task-4639`). Zero crashes; zero repeated inferences.

---

## Table 1: Standard Box-Level Default mAP vs. Legacy Image-Level AP

*Comparison of the submitted paper's claims against clean empirical re-evaluation on all $9,658$ test images.*

| Model & Evaluation Split | Paper Claimed (Image-Level) `[LOG]` | Measured Image-Level AP `[MEASURED]` | Measured Standard Box-Level Default mAP `[MEASURED]` |
|---|:---:|:---:|:---:|
| **Vynix 1.0 (YOLOv8m + CLIP ViT-B/16)** | | | |
| — Full ($600$ classes) | $30.31\%$ | **$30.25\%$** | **$12.23\%$** |
| — Standard Rare ($138$ classes) | — | $18.29\%$ | **$11.14\%$** |
| — Standard Non-Rare ($462$ classes) | — | $33.82\%$ | **$12.56\%$** |
| — Legacy Pseudo-Rare ($150$ classes, $h \bmod 4 == 0$) | $29.71\%$ | **$29.71\%$** | $11.87\%$ |
| — Legacy Pseudo-Non-Rare ($450$ classes) | $30.51\%$ | **$30.43\%$** | $12.35\%$ |
| **Vynix 2.0 (YOLOv8x + CLIP ViT-B/16)** | | | |
| — Full ($600$ classes) | $34.80\%$ | **$34.53\%$** | **$14.76\%$** |
| — Standard Rare ($138$ classes) | — | $24.96\%$ | **$15.37\%$** |
| — Standard Non-Rare ($462$ classes) | — | $37.39\%$ | **$14.58\%$** |
| — Legacy Pseudo-Rare ($150$ classes, $h \bmod 4 == 0$) | $36.10\%$ | **$35.96\%$** | $16.11\%$ |
| — Legacy Pseudo-Non-Rare ($450$ classes) | $34.37\%$ | **$34.05\%$** | $14.31\%$ |
| **Intermediate: YOLOv8x + Unaligned v1.0 Cache** | | | |
| — Full ($600$ classes) | $31.71\%$ | **$31.68\%$** | **$13.25\%$** |
| — Standard Rare ($138$ classes) | — | $19.77\%$ | **$12.02\%$** |
| — Standard Non-Rare ($462$ classes) | — | $35.23\%$ | **$13.62\%$** |
| — Legacy Pseudo-Rare ($150$ classes) | $30.63\%$ | **$30.69\%$** | $12.71\%$ |
| — Legacy Pseudo-Non-Rare ($450$ classes) | $32.07\%$ | **$32.01\%$** | $13.43\%$ |

> **Key Audit Finding `[LOG]` & `[MEASURED]`**:
> 1. The paper's claimed "Rare" and "Non-Rare" numbers were evaluated using a non-standard pseudo-rare heuristic (`h % 4 == 0`, $150$ classes) rather than the standard benchmark split ($138$ classes with $<10$ training instances). Under the pseudo-rare split, Vynix 1.0 achieves **$29.71\%$**, matching the paper's claim of $29.71\%$ to two decimal places.
> 2. When evaluated under the official standard HICO-DET Box-Level Default mAP protocol ($\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$), Vynix 1.0 achieves **$12.23\%$** Full mAP and Vynix 2.0 achieves **$14.76\%$** Full mAP (a $+2.53\%$ absolute gain / $+20.7\%$ relative gain).

---

## Table 2: Comprehensive Test-Set Ablation Suite

*All variants evaluated simultaneously in single test passes over all $9,658$ images (`[MEASURED]`). Deltas are relative to the Full model.*

### A. Vynix 1.0 Ablations (YOLOv8m detector, $\sigma=0.01, \lambda=1.00, \gamma_{\text{obj}}=0.0$)
| Ablation Configuration | Box Full mAP | $\Delta$ Box Full | Box Rare 138 | Box Non-Rare 462 | Img Full AP | $\Delta$ Img Full |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Pipeline** | **$12.23\%$** | **—** | **$11.14\%$** | **$12.56\%$** | **$30.25\%$** | **—** |
| w/o Geometric Gate ($\gamma_{\text{geo}} = 1.0$) | $10.34\%$ | **$-1.89\%$** | $10.27\%$ | $10.37\%$ | $29.86\%$ | $-0.39\%$ |
| Hard Binary Gate ($\text{IoU} > 0$) | $12.35\%$ | **$+0.12\%$** | $11.17\%$ | $12.70\%$ | $30.03\%$ | $-0.22\%$ |
| Raw Baseline (No Gate, $\lambda=1.0$) | $10.34\%$ | **$-1.89\%$** | $10.27\%$ | $10.37\%$ | $29.86\%$ | $-0.39\%$ |

### B. Vynix 2.0 Ablations (YOLOv8x detector, $\sigma=0.01, \lambda=0.90, \gamma_{\text{obj}}=0.10$)
| Ablation Configuration | Box Full mAP | $\Delta$ Box Full | Box Rare 138 | Box Non-Rare 462 | Img Full AP | $\Delta$ Img Full |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Pipeline** | **$14.76\%$** | **—** | **$15.37\%$** | **$14.58\%$** | **$34.53\%$** | **—** |
| w/o Geometric Gate ($\gamma_{\text{geo}} = 1.0$) | $12.53\%$ | **$-2.23\%$** | $14.28\%$ | $12.00\%$ | $34.38\%$ | $-0.15\%$ |
| Hard Binary Gate ($\text{IoU} > 0$) | $14.89\%$ | **$+0.13\%$** | $15.40\%$ | $14.74\%$ | $34.26\%$ | $-0.27\%$ |
| w/o Object Gate ($\gamma_{\text{obj}} = 0.0$) | $14.75\%$ | **$-0.01\%$** | $15.33\%$ | $14.58\%$ | $34.56\%$ | $+0.03\%$ |
| $\lambda = 1.00$ (detector power weighting) | $14.81\%$ | **$+0.05\%$** | $15.31\%$ | $14.66\%$ | $34.49\%$ | $-0.04\%$ |
| Raw Baseline (No Gates, $\lambda=1.0$) | $12.58\%$ | **$-2.18\%$** | $14.19\%$ | $12.10\%$ | $34.36\%$ | $-0.17\%$ |
| YOLOv8x + Unaligned v1.0 Cache | $13.25\%$ | **$-1.51\%$** | $12.02\%$ | $13.62\%$ | $31.68\%$ | $-2.85\%$ |

> **Key Ablation Finding `[MEASURED]`**:
> - The geometric veto gate provides the single largest performance contribution in the entire pipeline: **$+1.89\%$ Box mAP** on Vynix 1.0 and **$+2.23\%$ Box mAP** on Vynix 2.0.
> - The Object Semantic Gate ($\gamma_{\text{obj}}=0.10$) and detector calibration ($\lambda=0.90$) have near-neutral impacts on Box-level mAP ($\le 0.05\%$).
> - Retraining the cache and spatial MLP on native detector proposals is critical: pairing YOLOv8x proposals with the unaligned v1.0 cache drops Box mAP by **$-1.51\%$** ($14.76\% \to 13.25\%$).

---

## Table 3: Hypothesis-Level Veto and Rescue Audit

*Exact counts across all candidate hypotheses generated across all $9,658$ test images (`[MEASURED]`).*

### Exact Operational Definitions (`[LOG]`):
1. **Candidate Hypothesis**: A detected `(person, object)` proposal pair within the top-50 pairs per image, mapped to a valid HICO-DET object class, and expanded across valid verbs for that object.
2. **Contact Verb**: Predicate where physical contact is required (e.g., `hold`, `sit_on`, `ride`, `eat`).
3. **Soft Veto**: Condition `gate < 0.05` under the continuous spatial attenuation function $\exp(-(d_{\text{box}} / \sigma)^2)$.
4. **Hard Binary Veto**: Condition $\text{IoU} = 0.0$ for any contact verb.
5. **Box-Level Matched GT**: Candidate triplet matches an annotated ground truth of the same HOI class with $\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$.
6. **Image-Level Matched GT**: Candidate HOI class is present in the image's ground-truth positive set.
7. **Unmatched False Positive**: Candidate hypothesis does not match any ground-truth annotation for the image.
8. **Rescued Hypothesis**: Contact interaction with non-overlapping bounding boxes ($\text{IoU}=0.0$) but near-touching distance ($d_{\text{box}} \le \sigma$) that is verified to match ground truth.

| Hypothesis Metric | Paper Claim `[LOG]` | Vynix 1.0 (YOLOv8m) `[MEASURED]` | Vynix 2.0 (YOLOv8x) `[MEASURED]` |
|---|:---:|:---:|:---:|
| **Total Candidate Hypotheses** | — | $1,077,751$ | $1,108,180$ |
| — Contact Predicate Hypotheses | — | $635,082$ ($58.9\%$) | $653,861$ ($59.0\%$) |
| — Non-Contact Predicate Hypotheses | — | $442,669$ ($41.1\%$) | $454,319$ ($41.0\%$) |
| **Soft Gate Vetoes (Nominal $\sigma=0.08$)** | `265,189` / `272,946` | **$265,189$** (exact match) | **$272,946$** (exact match) |
| — Box-Level Matched GT (True Positives Suppressed) | — | $215$ ($0.08\%$) | $228$ ($0.08\%$) |
| — Image-Level Matched GT | — | $36,106$ ($13.6\%$) | $37,549$ ($13.8\%$) |
| — Unmatched False Positives Suppressed | — | **$229,083$** ($86.4\%$) | **$235,397$** ($86.2\%$) |
| **Soft Gate Vetoes (Frozen $\sigma=0.01$)** | — | **$424,343$** | **$436,338$** |
| — Box-Level Matched GT | — | $538$ ($0.13\%$) | $570$ ($0.13\%$) |
| — Image-Level Matched GT | — | $58,848$ ($13.9\%$) | $60,807$ ($13.9\%$) |
| — Unmatched False Positives Suppressed | — | **$365,495$** ($86.1\%$) | **$375,531$** ($86.1\%$) |
| **Hard Binary Gate Vetoes ($\text{IoU}=0$)** | `319,803` / `463,346` | **$450,498$** | **$463,346$** (exact match) |
| — Box-Level Matched GT | — | $644$ ($0.14\%$) | $690$ ($0.15\%$) |
| — Image-Level Matched GT | — | $62,630$ ($13.9\%$) | $64,687$ ($14.0\%$) |
| — Unmatched False Positives Suppressed | — | **$387,868$** ($86.1\%$) | **$398,659$** ($86.0\%$) |
| **Additional Hard Vetoes over Nominal Soft Gate** | `190,400` (v2.0) | $185,309$ | **$190,400$** (exact match) |
| **Rescued Contact Hypotheses ($\text{IoU}=0, d_{\text{box}} \le \sigma$)** | `54,532` | | |
| — Under Nominal $\sigma=0.08$ (Box-Matched GT) | — | **$295$** | **$327$** |
| — Under Nominal $\sigma=0.08$ (Image-Matched GT) | — | **$16,153$** | **$16,431$** |
| — Under Frozen $\sigma=0.01$ (Box-Matched GT) | — | **$65$** | **$75$** |
| — Under Frozen $\sigma=0.01$ (Image-Matched GT) | — | **$2,189$** | **$2,319$** |

> **Key Veto & Rescue Audit Finding `[LOG]` & `[MEASURED]`**:
> 1. **Veto Accuracy**: The paper's claimed veto counts ($265,189$ for v1.0, $272,946$ for v2.0, $463,346$ for hard binary gate, and $190,400$ additional vetoes) are replicated **to the exact single hypothesis**.
> 2. **Mechanism Validation**: **$86.1\% - 86.4\%$** of all vetoed hypotheses are verified false positives (unmatched to any GT annotation). Only $0.08\% - 0.15\%$ of vetoed pairs are box-matched ground truths.
> 3. **The "54,532 Rescued Interactions" Claim**: In the original manuscript, "54,532 rescued interactions" represented the total hypothesis count reduction ($319,803 - 265,189 = 54,614$, subject to an 82-count arithmetic discrepancy in the original text). In actual ground-truth data, the number of true positive contact interactions with $\text{IoU}=0$ rescued by the soft gate is **$295$** at box level ($16,153$ at image level) for Vynix 1.0, and **$327$** at box level ($16,431$ at image level) for Vynix 2.0.

---

## Table 4: Soft Gate ($\sigma=0.01$) vs. Hard Binary Gate on TEST Set

*Direct comparison of continuous geometric attenuation versus binary thresholding on both evaluation protocols (`[MEASURED]`).*

| Model & Gate Variant | Standard Box-Level Default mAP | Legacy Image-Level AP | Total Vetoes Suppressed | True Positive Veto Penalty (Box Matched) |
|---|:---:|:---:|:---:|:---:|
| **Vynix 1.0** | | | | |
| — Soft Geometric Gate ($\sigma=0.01$) | $12.23\%$ | **$30.25\%$** | $424,343$ | $538$ |
| — Hard Binary Gate ($\text{IoU} > 0$) | **$12.35\%$** | $30.03\%$ | $450,498$ | $644$ |
| — *Difference ($\text{Soft} - \text{Hard}$)* | *$-0.12\%$* | **$+0.22\%$** | *$-26,155$* | *$-106$* |
| **Vynix 2.0** | | | | |
| — Soft Geometric Gate ($\sigma=0.01$) | $14.76\%$ | **$34.53\%$** | $436,338$ | $570$ |
| — Hard Binary Gate ($\text{IoU} > 0$) | **$14.89\%$** | $34.26\%$ | $463,346$ | $690$ |
| — *Difference ($\text{Soft} - \text{Hard}$)* | *$-0.13\%$* | **$+0.27\%$** | *$-27,008$* | *$-120$* |

> **Divergence Explanation `[MEASURED]`**:
> On the legacy image-level AP metric, soft continuous attenuation outperforms hard binary vetoing by **$+0.22\%$ to $+0.27\%$** because borderline contact interactions retain partial confidence and contribute to average precision. On the box-level Default protocol ($\min(\text{IoU}_h, \text{IoU}_o) \ge 0.5$), non-overlapping boxes ($\text{IoU}=0$) are almost never true positives, giving the hard binary gate a marginal $+0.12\% - 0.13\%$ edge.

---

## Table 5: Stage-Wise Wall-Clock Timing & Compute Budget

*Breakdown of inference latency across all $9,658$ test images measured on an RTX 3050 Ti laptop GPU (`[MEASURED]`).*

| Pipeline Stage | Vynix 1.0 (YOLOv8m) Total | Vynix 1.0 Per-Image | Vynix 1.0 Share | Vynix 2.0 (YOLOv8x) Total | Vynix 2.0 Per-Image | Vynix 2.0 Share |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Object Detection (YOLO)** | $5.60\text{ min}$ | $34.8\text{ ms}$ | $5.8\%$ | $378.15\text{ min}$ | $2,349.2\text{ ms}$ | $80.1\%$ |
| **2. Feature Encoding (CLIP ViT-B/16)** | $89.68\text{ min}$ | $557.1\text{ ms}$ | $92.9\%$ | $92.06\text{ min}$ | $571.9\text{ ms}$ | $19.5\%$ |
| **3. Adapter & Gate Rescoring** | $0.74\text{ min}$ | $4.6\text{ ms}$ | $0.8\%$ | $1.22\text{ min}$ | $7.6\text{ ms}$ | $0.3\%$ |
| **4. Evaluation Overhead** | $0.10\text{ min}$ | $0.6\text{ ms}$ | $0.1\%$ | $0.20\text{ min}$ | $1.3\text{ ms}$ | $0.0\%$ |
| **Total Test Wall-Clock** | **$96.58\text{ min}$** | **$600.0\text{ ms}$** | **$100.0\%$** | **$472.17\text{ min}$** | **$2,933.4\text{ ms}$** | **$100.0\%$** |
| **Effective Inference Throughput** | **$1.67\text{ img/s}$** | — | — | **$0.34\text{ img/s}$** | — | — |

> **Training Time vs. Test Time Explanation (Answers Reviewer B.1) `[LOG]` & `[MEASURED]`**:
> - **Why Training Takes ~17–19 Minutes**: Training uses class-balanced prototype capping (`max_shots_per_class = 50`, `train_vynix_full.py:24, 247`). Once all HOI categories present in a training image have reached 50 exemplars, the entire image is skipped before YOLO detection and before CLIP feature extraction (`train_vynix_full.py:80-84`). On `train_90`, this bypassed $28,379$ out of $34,306$ training images ($82.7\%$). Only $\sim 5,700 - 5,900$ images were processed, caching $\sim 19,600$ pairs.
> - **Why Test Takes ~96–472 Minutes**: In test inference, zero images are skipped: all $9,658$ images are fully detected and up to 50 pairs ($150$ image crops) per image are encoded by CLIP. Furthermore, YOLOv8x inference without TensorRT/FP16 on a laptop GPU takes $\sim 2.35$ seconds per image, accounting for $80.1\%$ of Vynix 2.0's test latency.

---

## Table 6: Test Ground-Truth Class Coverage

*Verification of class representation in the $9,658$ HICO-DET test images (`[MEASURED]`).*

| Evaluation Category | Benchmark Classes | Classes with $\ge 1$ Test GT Instance | Coverage | Total Annotated GT Instances |
|---|:---:|:---:|:---:|:---:|
| **Full Category Set** | $600$ | $600$ | **$100.0\%$** | $33,405$ |
| **Standard Rare Split** ($<10$ train instances) | $138$ | $138$ | **$100.0\%$** | $846$ |
| **Standard Non-Rare Split** ($\ge 10$ train instances) | $462$ | $462$ | **$100.0\%$** | $32,559$ |

---

## Complete Methodological Disclosures (`DISCLOSURES.md` Summary)

1. **Exploratory Test-Set Evaluations (`eval_limit=300`)**: During early prototype development, interactive test subsets were run to benchmark latency and verify pipeline integration before the formal held-out validation protocol on `val_10` was created. Preliminary values of $\lambda=0.90$ and $\gamma_{\text{obj}}=0.10$ were informed by these exploratory runs.
2. **Text Prompt Ensembling**: The 8-template prompt ensemble for positive interactions and 4 negative interaction templates were developed and tuned based on observing interaction disambiguation during early system development.
3. **Historical $\sigma=0.08$ Provenance**: The value $\sigma=0.08$ in the submitted manuscript was selected from an initial test-set sweep. Under the clean validation split (`val_10`) using standard Box-Level Default mAP, $\sigma=0.01$ was selected by the pre-committed selection rule.
4. **Detector Proposal Confidence Threshold ($\tau_{\text{det}}$)**: Although the manuscript stated proposals were filtered at $\tau_{\text{det}} = 0.08$, Ultralytics YOLO defaults to `conf=0.25`. Because `conf` was not overridden in `MultiStreamFeatureExtractor.detect()`, all proposals generated throughout training, caching, and test evaluation had confidence $\ge 0.25$ (averaging $\sim 16.9 - 17.5$ pairs per image). The planned $\tau_{\text{det}} \in \{0.05, 0.08, 0.12\}$ sweep was mechanically inactive.
5. **Cache Saturation (50-Shot Cap, Not 16-Shot)**: Cache extraction enforces `max_shots_per_class = 50` (`train_vynix_full.py:24, 247`). Once all HOI categories reach 50 positive exemplars, subsequent training images are completely skipped. On `train_90`, 28,379 images were bypassed, constructing the cache from 5,927 images ($19,579$ pairs). Zero images are bypassed at test time.
