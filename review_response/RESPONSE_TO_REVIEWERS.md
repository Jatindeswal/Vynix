# Draft Response to Reviewers (`RESPONSE_TO_REVIEWERS.md`)

> **Note to Authors:** Brackets `[TODO: ...]` mark numbers that are currently `[UNKNOWN]` because script execution was blocked by the harness CWD mismatch and/or requires your approval for >30-min GPU runs. Do **not** submit until those runs complete or the `[TODO]`s are filled with `[MEASURED]` outputs.

---

## Response to Reviewer A

**1. Novelty vs. Existing Frozen-Detector + Spatial Frameworks (UPT, SCG):**
We thank the reviewer for highlighting SCG (Zhang et al., ICCV 2021) and UPT (Zhang et al., CVPR 2022). Both SCG and UPT pair a frozen detector (Faster R-CNN / DETR) with pairwise geometric encodings inside learned graph/transformer heads trained over all pairs. Vynix differs in two ways: (i) both the detector and CLIP ViT-B/16 encoder remain frozen, with training restricted to a lightweight `SpatialMLP` (`8 -> 64 -> 600`) and scalar cache parameters (`alpha, beta`) on a 50-shot/class exemplar cache; and (ii) we introduce an explicit, training-free predicate-conditioned Gaussian veto gate ($\gamma$) that attenuates zero-overlap contact hypotheses without suppressing non-contact predicates. We have added this discussion to Section II (`PAPER_PATCHES.md`, Patch 4).

**2. Validating the Geometric Veto Mechanism:**
On the full 9,658-image HICO-DET evaluation of Vynix 2.0 (`full_test_ablation_results.json` `[LOG]`), the soft gate ($\sigma = 0.08$) improves Full mAP from `34.53%` (no gate) to `34.80%`, whereas a hard binary gate ($\text{IoU} > 0$) drops performance to `34.33%` (`-0.48%` vs. soft gate) due to `190,400` extra vetoes on border-touching interactions. `[TODO: Insert clean val_10 sigma sensitivity, contact vs. non-contact mAP breakdown, and hypothesis-level TP-lost vs. FP-suppressed counts from Task 4 & Task 6]`.

**3. Contact vs. Non-Contact Predicates:**
We now provide the complete partition of all 117 HICO-DET verbs (`05_predicates.csv` `[LOG]`): `59` HICO-DET verbs are gated as physical-contact predicates (out of `64` strings in `CONTACT_VERBS`, 5 of which—*grab, mount, play_with, taste, touch*—do not appear as canonical HICO-DET verb labels), while `58` predicates (including *watch, point, wave*, and `no_interaction`) unconditionally receive $\gamma = 1.0$. We also acknowledge borderline cases (e.g., *fly* for kite vs. airplane; *hose* via water stream; and tool-contact verbs like *paint, repair, stir* currently excluded) as a limitation and avenue for object-conditioned gating.

**4. Cross-Dataset Validation (V-COCO):**
We agree that cross-dataset evaluation on V-COCO (`AP_role`) is valuable. V-COCO is not included in the current repository release (`08_cross_dataset.md`), and we outline the exact adaptation plan (26-action `VCOCOMeta` + `vsrl_eval` protocol, ~15 min cache + ~45 min test on RTX 3050 Ti). `[TODO: State whether V-COCO was executed for camera-ready or deferred to future work.]`

---

## Response to Reviewer B

**B.1 Training Time (17–19 min) vs. Test Time (83–92 min) & Feature Caching:**
Yes, disk-level shard caching (`cache_shard_XX.pt`, `vynix_full_cache.pt`) and class-balanced exemplar capping (`max_shots_per_class = 50`, `<= 20` matching pairs/image) are used during training (`train_vynix_full.py` `[LOG]`). Once all HOI classes present in a training image reach 50 positive exemplars, both YOLO and CLIP inference are skipped for that image. Consequently, training processes only a sparse positive subset (`[TODO: insert measured N_train_processed and N_cached_pairs]`), taking `17.33 min` (Vynix 1.0) and `19.35 min` (`17.33 min` extraction + `2.02 min` 15-epoch MLP training, Vynix 2.0). At test time, all `9,658` images are evaluated without skipping, encoding up to `50` pairs (`150` CLIP crops) per image at batch size $\le 8$, taking `83.49 min` (Vynix 1.0) and `91.74 min` (Vynix 2.0).

**B.2 & B.5 Validation Split & Hyperparameter Tuning:**
Our audit confirmed that Vynix 2.0's $\lambda = 0.90$ and $\gamma_{\text{obj}} = 0.10$ were previously checked on a 300-image slice of the test set (`--limit 300`), with no train-derived validation split. To completely eliminate test-set leakage, we constructed a deterministic 90/10 split of the 38,118 HICO-DET training images (`34,306` train / `3,812` validation, seed `42`), tuned hyperparameters coordinate-wise on the validation split only, and evaluated the frozen configuration once on the test set (`[TODO: insert Task 3 val & frozen test table]`).

**B.3 Rescued & Vetoed Pair Counts:**
We audited the veto counters (`gate < 0.05` in `test_vynix_pretrained.py:144` and `run_full_test_ablation.py:291` `[LOG]`). For Vynix 2.0, `463,346` (hard gate) minus `272,946` (soft gate) equals `190,400` extra vetoes, matching our logs. For Vynix 1.0, `319,803` (hard) minus `265,189` (soft) equals `54,614` rescued border-touching hypotheses (correcting a `54,532` typographical discrepancy of 82 in the draft). `[TODO: Insert TP-lost vs. FP-suppressed split from audit_vetoes.py.]`
