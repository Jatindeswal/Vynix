# Paper Patches (`PAPER_PATCHES.md`)

All patches below replace inaccurate claims with code-verified `[LOG]` facts. Items marked `[TODO: ...]` require running the corresponding verification script once the execution environment is unblocked.

---

## Patch 1 — Architecture & Scoring Formula (Fixes MLP, Cache Dim, $\alpha$, and Gate Formula)

**Replace** the current cache/MLP/gate equations with:
> For each candidate `(person, object)` pair, frozen CLIP ViT-B/16 encodes 3 crops (human, object, union) into L2-normalized $f_v \in \mathbb{R}^{1536}$, which serves directly as the visual cache key $k_e \in \mathbb{R}^{1536}$. Simultaneously, an 8-D relative geometry vector $g$ is mapped by a 2-layer `SpatialMLP` (`Linear(8,64)-LayerNorm-ReLU-Linear(64,600)-Sigmoid`) to class-wise spatial log-priors. The adapter learns scalar parameters $\alpha$ (initialized to $1.0$) and $\beta$ (initialized to $5.5$) and computes class logits as $\text{logits} = S_{\text{vlm}} + \alpha \cdot \exp(-\beta(1 - f_v K_{\text{cache}}^T)) V_{\text{cache}} + 0.5 \log(P_{\text{spatial}} + 10^{-6})$. Final pair-verb confidence is $S_{\text{HOI}} = M_{c_o,a} (s_h s_o)^\lambda G_{\text{obj}}(o) \gamma(a, \text{IoU}, d_{\text{box}}) \cdot \text{softmax}(\text{logits})_a$, where $\gamma = \exp(-d_{\text{box}}^2 / \sigma^2)$ ($\sigma = 0.08$) for `IoU = 0` contact verbs and $1.0$ otherwise.

---

## Patch 2 — Training Time & Feature Caching (Answers Reviewer B.1)

> Training Vynix does not extract CLIP features for all candidate pairs across all 38,118 training images; instead, `build_full_dataset_cache` caps the non-parametric memory at at most 50 positive exemplars per HOI class (and at most 20 matching pairs per image), skipping both YOLO and CLIP inference for any image whose present classes are already saturated and caching the resulting 1536-D embeddings to disk shards. Because this cap reduces training extraction to a sparse, class-balanced subset (`[TODO: insert measured N_train_processed images and N_cached_pairs]` `[UNKNOWN]`), full cache construction plus 15 epochs of adapter/MLP training takes 17.33 min (Vynix 1.0) or 19.35 min (17.33 min extraction + 2.02 min MLP training, Vynix 2.0) on an RTX 3050 Ti (4 GB VRAM). By contrast, test evaluation runs YOLO and 3-stream CLIP ViT-B/16 (batch size $\le 8$) on all 9,658 test images with up to 50 candidate pairs (150 crops) per image and no class-saturation skipping, resulting in 83.49 min (Vynix 1.0) and 91.74 min (Vynix 2.0) end-to-end inference time.

---

## Patch 3 — Contact vs. Non-Contact Predicates (Answers Reviewer A)

> We partition the 117 HICO-DET verb predicates into a manually curated set of 59 active physical-contact predicates (e.g., *hold, carry, ride, sit_on, eat, cut*, out of 64 entries in `CONTACT_VERBS`, 5 of which—*grab, mount, play_with, taste, touch*—do not appear as canonical HICO-DET verb labels) and 58 non-contact or unconstrained predicates (including visual/gestural verbs such as *watch, point, wave* and `no_interaction`). Non-contact predicates bypass spatial attenuation (`gamma = 1.0` unconditionally), whereas contact predicates are attenuated by the soft Gaussian distance gate whenever the human and object boxes are disjoint (`IoU = 0`).

---

## Patch 4 — Related Work: SCG & UPT (Answers Reviewer B.4 & Reviewer A)

> Closely related to our decoupled design, two-stage HOI detectors such as **SCG** (Zhang et al., ICCV 2021) and **UPT** (Zhang et al., CVPR 2022) also pair a frozen off-the-shelf object detector (Faster R-CNN or DETR) with explicit pairwise geometric encodings (relative displacements, scale ratios, and IoU) to disambiguate interactions. Whereas SCG and UPT train multi-layer graph or transformer heads over detector RoI/query features to learn spatial compatibility implicitly across all candidate pairs, Vynix keeps both the detector and CLIP ViT-B/16 vision-language encoder frozen, restricts learned spatial modeling to a lightweight 2-layer MLP (`SpatialMLP`) trained on a class-balanced exemplar cache, and introduces an explicit, training-free predicate-conditioned Gaussian veto gate ($\gamma$) that attenuates zero-overlap contact hypotheses.

---

## Patch 5 — Evaluation Protocol & Veto/Rescue Counts (Mandatory Integrity Fix)

- **Metric & Split Disclosure:** Either re-evaluate using official HICO-DET box-level Default mAP ($\text{IoU} \ge 0.5$) + official 155/445 Rare/Non-Rare split, OR explicitly state in Section IV that the reported mAPs are **image-level multi-label AP** with a 150-class (`h % 4 == 0`) pseudo-rare split and **cannot** be compared numerically to box-level SOTA tables.
- **Rescued Count Fix:** Replace `54,532` rescued contacts with `54,614` (`319,803 - 265,189`, `[LOG]`) or update after running `review_response/audit_vetoes.py`.
- **Clean Validation Results:** Insert `[TODO: val_10 & frozen test results from Task 3]` once Task 3 is approved and executed.
