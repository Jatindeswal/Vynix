# Task 7 — Related Work Comparison: SCG and UPT (Answers Reviewer B.4 & Reviewer A)

## 1. Literature Verification (`[LOG]` + Web Search)

1. **SCG — *Spatially Conditioned Graphs for Detecting Human-Object Interactions* (Zhang, Campbell, & Gould, ICCV 2021):**
   - Uses an off-the-shelf COCO-pretrained **Faster R-CNN** (frozen detector) to propose human/object boxes, then constructs a bipartite graph where message passing between human and object nodes is **conditioned on pairwise spatial encodings** (box centers, widths/heights, area ratios, IoU, relative direction).
   - **HICO-DET Default mAP:** `31.33%` (with fine-tuned DRG detections); `29.26%` (`24.61%` Rare / `30.65%` Non-Rare with DETR-R101).
2. **UPT — *Efficient Two-Stage Detection of Human-Object Interactions with a Novel Unary-Pairwise Transformer* (Zhang, Campbell, & Gould, CVPR 2022):**
   - Uses a **frozen / two-stage DETR** detector to extract unary human/object tokens, followed by cooperative/competitive unary attention and pairwise spatial positional encodings (including box sizes, area ratios, IoU, and directional `ReLU(d)` / `ReLU(-d)` terms) fed into an MLP and transformer layers.
   - **HICO-DET Default mAP:** `31.66%` Full / `25.94%` Rare / `33.36%` Non-Rare (`UPT-R50`).

---

## 2. Conceptual & Architectural Comparison

| Dimension | SCG (ICCV 2021) | UPT (CVPR 2022) | Vynix (Ours) |
|---|---|---|---|
| **Frozen Detector** | Faster R-CNN (ResNet-FPN) | DETR (ResNet-50 / 101) | YOLOv8m / YOLOv8x |
| **Visual Representation** | RoI-pooled detector backbone features | DETR decoder object queries + ResNet features | Frozen CLIP ViT-B/16 3-stream crops (`f_h, f_o, f_u`) |
| **How Pairwise Spatial Features Are Used** | Conditions graph edge weights & message passing | Pairwise positional encodings inside Transformer layers | 8-D geometry -> `SpatialMLP` (`8->64->600`) log-prior added to logits (`vynix_fewshot_adapter.py:249-266, 561-562`) |
| **Training Scope** | Trains graph network + classification heads on all pairs | Trains unary + pairwise transformer layers on all pairs | Trains only lightweight `SpatialMLP` + scalar `alpha, beta` + cache keys on `<=50` positive shots/class |
| **Explicit Geometric Veto** | None (learned soft spatial conditioning only) | None (learned spatial attention only) | Explicit predicate-conditioned Gaussian distance veto $\gamma(a, \text{IoU}, d_{\text{box}})$ |
| **Evaluation Caveat (`[LOG]`)** | Official HICO-DET **box-level** Default mAP ($\text{IoU} \ge 0.5$) | Official HICO-DET **box-level** Default mAP ($\text{IoU} \ge 0.5$) | Repo scripts evaluate **image-level** AP + `h % 4 == 0` pseudo-rare split — **NOT directly comparable** to SCG/UPT box-level mAP without re-running box-level eval! |

---

## 3. Ready-to-Paste Related Work Paragraph (Honest Positioning)

> Closely related to our decoupled design, two-stage HOI detectors such as **SCG** (Zhang et al., ICCV 2021) and **UPT** (Zhang et al., CVPR 2022) also pair a frozen off-the-shelf object detector (Faster R-CNN or DETR) with explicit pairwise geometric encodings (relative displacements, scale ratios, and IoU) to disambiguate interactions. Whereas SCG and UPT train multi-layer graph or transformer heads over detector RoI/query features to learn spatial compatibility implicitly across all candidate pairs, Vynix keeps both the detector and CLIP ViT-B/16 vision-language encoder frozen, restricts learned spatial modeling to a lightweight 2-layer MLP (`SpatialMLP`) trained on a class-balanced exemplar cache, and introduces an explicit, training-free predicate-conditioned Gaussian veto gate ($\gamma$) that attenuates zero-overlap contact hypotheses. *(Note for authors: Because Vynix's current evaluation scripts measure image-level AP rather than HICO-DET box-level IoU $\ge 0.5$ mAP, direct numerical comparison against SCG's `31.33%` and UPT's `31.66%` box-level mAP must be withheld or recomputed under identical box-level HICO-DET evaluation.)*
