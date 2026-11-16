# Vynix Reproducibility & Peer-Review Audit Report

**Repository:** Vynix (`vynix_fewshot_adapter.py`, `train_vynix_full.py`, `test_vynix_pretrained.py`, `run_full_test_ablation.py`)  
**Hardware (logged):** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (4 GB VRAM) (`[LOG]` `process.md:105, 346`)  
**Git Branch / Commit Hash / Library Versions:** `[UNKNOWN]` — `run_command` is blocked in this session because the harness attempts to start in a non-existent drive (`e:\...` instead of the actual repository drive). Therefore, `git checkout -b review-response`, `git log -p`, and live Python measurements could not be executed.

---

## 1. Executive Summary of Critical Contradictions (`[LOG]`)

Before submitting any rebuttal, the authors must resolve **7 concrete discrepancies** between the paper draft and the actual codebase/logs:

1. **Evaluation Metric Is Image-Level AP, NOT Box-Level HICO-DET Default mAP (`[LOG]`):**
   - *Paper claims:* HICO-DET Default setting (both human and object boxes $\text{IoU} \ge 0.5$).
   - *Code implements (`vynix_fewshot_adapter.py:777, 834-844`; `test_vynix_pretrained.py:165-170`; `run_full_test_ablation.py:57-64`):* Image-level multi-label AP (`all_gt[img_idx] = set(positive_objects)`, `all_preds[img_idx][hoi_id] = max over pairs`). **No bounding-box IoU matching against ground-truth boxes is performed.** Consequently, reported mAPs (`30.31%`, `34.80%`) are **not** directly comparable to standard box-level HICO-DET SOTA tables (SCG, UPT, GEN-VLKT, etc.).
2. **Rare / Non-Rare Split Is a 150-Class Pseudo-Split, NOT Official 155/445 (`[LOG]`):**
   - *Paper claims:* 155 Rare ($<10$ training instances) / 445 Non-Rare.
   - *Code implements (`vynix_fewshot_adapter.py:847`; `test_vynix_pretrained.py:173`; `run_full_test_ablation.py:66`):* `rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}` (150 classes = every 4th index).
3. **Soft Geometric Gate Missing Factor of 2 in Exponent (`[LOG]`):**
   - *Paper claims:* $\exp(-d_{\text{box}}^2 / (2\sigma^2))$ with $\sigma = 0.08$.
   - *Code implements (`vynix_fewshot_adapter.py:187`):* `math.exp(-(b_dist / max(sigma_contact, 1e-4))**2)` ($\exp(-d_{\text{box}}^2 / \sigma^2)$), which is equivalent to $\sigma_{\text{paper}} = 0.08 / \sqrt{2} \approx 0.0566$. (Also note `process.md:349` logs $\sigma_{\text{contact}} = 0.15$ for Phase 8, while the code hardcodes `0.08`).
4. **Spatial MLP & Cache Key Architecture Mismatch (`[LOG]`):**
   - *Paper claims:* `Linear(8,128)-GELU-Linear(128,256)-GELU`, concatenated into cache key $k_e = [f_v; f_{\text{geom}}] \in \mathbb{R}^{1792}$.
   - *Code implements (`vynix_fewshot_adapter.py:249-266, 523-562`):* `SpatialMLP` is `Linear(8, 64)-LayerNorm(64)-ReLU-Linear(64, 600)-Sigmoid` added as `+ 0.5 * log(spatial_priors + 1e-6)` to logits; `cache_keys` is $\mathbb{R}^{1536}$ (3-stream visual only, no geometry concatenation).
5. **Blending Weight $\alpha$ Is Learned (Init `1.0`) & Additive, NOT Fixed `0.35` Convex (`[LOG]`):**
   - *Paper claims:* Fixed $\alpha = 0.35$ in $(1 - \alpha) S_{\text{vlm}} + \alpha S_{\text{cache}}$.
   - *Code implements (`vynix_fewshot_adapter.py:524-525, 556`):* `self.alpha = nn.Parameter(torch.tensor(1.0))` and `self.beta = nn.Parameter(torch.tensor(5.5))` are **learned** via AdamW, and blended additively as `clip_logits + alpha * cache_logits`. (`process.md:332` confirms overriding learned $\alpha$ degraded mAP.)
6. **Test-Set Tuning Exists (`[LOG]`):**
   - `process.md:318-329` and `run_option1_retrain_and_eval.py:59` explicitly tuned $\lambda = 0.90$, $\gamma_{\text{obj}} = 0.10$, prompt ensembling, and adaptive-$\alpha$ on the first **300 images of the test set** (`--limit 300`). No train-derived validation split exists.
7. **Veto / Rescue Arithmetic & Contact Verb Count (`[LOG]`):**
   - Vynix 1.0 hard vetoes (`319,803`) minus soft vetoes (`265,189`) = `54,614`, which differs by **82** from the paper's claimed `54,532` rescued contacts.
   - `CONTACT_VERBS` (`vynix_fewshot_adapter.py:79-88`) has `64` strings, but `5` (`grab, mount, play_with, taste, touch`) do not appear as `vname` in `list_action.csv`; thus `59` of `117` HICO-DET verbs are gated and `58` are non-contact.

---

## 2. Sub-Report Index

- [00_repo_map.md](00_repo_map.md) — Phase 0 entry points, mechanism locations, and test-run inventory.
- [01_training_time.md](01_training_time.md) & [01_stage_times.csv](01_stage_times.csv) — Task 1 training-time audit (`max_shots=50` class-saturation skipping + disk sharding).
- [02_hyperparameter_provenance.md](02_hyperparameter_provenance.md) — Task 2 hyperparameter provenance & 9-run test-eval log.
- [03_validation_protocol.md](03_validation_protocol.md) — Task 3 clean 90/10 train-derived validation protocol (awaiting approval + CWD fix).
- [04_veto_audit.md](04_veto_audit.md) & [audit_vetoes.py](audit_vetoes.py) — Task 4 veto/rescue definitions, arithmetic check, and verification script.
- [05_predicates.md](05_predicates.md) & [05_predicates.csv](05_predicates.csv) — Task 5 complete 117-verb audit.
- [06_veto_validation.md](06_veto_validation.md) — Task 6 gate ablation audit & planned `val_10` breakdown.
- [07_related_work.md](07_related_work.md) — Task 7 SCG (ICCV '21) & UPT (CVPR '22) comparison.
- [08_cross_dataset.md](08_cross_dataset.md) — Task 8 V-COCO feasibility & effort estimate.
