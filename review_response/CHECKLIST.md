# Author Rebuttal Action Checklist (`CHECKLIST.md`)

- [ ] **Fix Execution Environment:** Ensure shell CWD points to the actual repository drive (`D:\Do not open\Projects\Vynix`) and create branch `git checkout -b review-response`.
- [ ] **Resolve Critical Paper-vs-Code Contradictions (`REPORT.md` §1):**
  - [ ] Decide whether to implement true **box-level HICO-DET Default mAP** ($\text{IoU} \ge 0.5$) + official **155/445 Rare/Non-Rare split** (requiring GT box coordinates from HICO-DET annotations if not in parquet), OR transparently re-label current numbers as **image-level multi-label AP** with a 150-class (`h % 4 == 0`) split and remove direct numerical comparisons to box-level SOTA tables.
  - [ ] Apply `PAPER_PATCHES.md` Patch 1 (exact `SpatialMLP` 8->64->600, 1536-D cache keys, learned `alpha`/`beta` additive blending, and `exp(-(d/sigma)^2)` gate formula).
  - [ ] Fix `54,532` -> `54,614` rescued count typo (`319,803 - 265,189`).
- [ ] **Run Task 1 & Task 4 Verification Scripts (Estimated ~15–20 min on RTX 3050 Ti for 500-img timing; ~90 min for full `audit_vetoes.py`):**
  - [ ] Inspect `saved_models_v1.0/vynix_full_cache.pt` and `saved_models_v2.0/vynix_full_cache.pt` shapes, learned `alpha`/`beta` values, and exact `N_cached_pairs`.
  - [ ] Run `python review_response/audit_vetoes.py --dataset-dir "E:\Dataset"` to populate `04_veto_audit_results.csv` (TP-lost vs FP-suppressed).
- [ ] **Approve & Run Task 3 Clean Validation Protocol (`03_validation_protocol.md`):**
  - [ ] Approve ~4.5 hours total GPU budget on RTX 3050 Ti (`train_90` cache+MLP, `val_10` feature extraction + grid search, single frozen test eval for Vynix 1.0 & 2.0).
  - [ ] Fill `[TODO]` placeholders in `RESPONSE_TO_REVIEWERS.md` and `PAPER_PATCHES.md` with `[MEASURED]` outputs.
