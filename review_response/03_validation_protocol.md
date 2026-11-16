# Task 3 — Clean Validation Protocol (Proposed Plan — Awaiting Approval)

**Why Required:** Task 2 (`02_hyperparameter_provenance.md`) confirmed `[LOG]` that $\lambda$, $\gamma_{\text{obj}}$, and adaptive-$\alpha$ were selected on a 300-image slice of the HICO-DET **test set**, and $\sigma, \alpha, \beta, s_{\text{conf}}$ were ablated on the test set. No train-derived validation split exists.

**Hardware & Runtime Gate:** Because full-split feature extraction and multi-grid validation sweeps on an RTX 3050 Ti Laptop GPU (~4 GB VRAM) will exceed 30 minutes, **no heavy job has been launched**. Execution also requires resolving the `run_command` harness CWD mismatch (`e:\...` vs actual repo drive).

---

## 1. Proposed Deterministic Split & Protocol

1. **Deterministic 90/10 Train/Val Split (Seed `42`):**
   - Index all `38,118` images across `train-*.parquet` in sorted shard/row order.
   - Shuffle indices with `np.random.RandomState(42)` and partition:
     - `train_90_ids.json`: `34,306` images (90%) — used **exclusively** for building the cache (`max_shots=50`) and training `SpatialMLP` + `alpha`/`beta` (15 epochs).
     - `val_10_ids.json`: `3,812` images (10%) — used **exclusively** for coordinate-wise hyperparameter tuning.
2. **Coordinate-Wise Coarse Grid Search (on `val_10` only):**
   - Pre-extract and cache `val_10` YOLO proposals + 3-stream CLIP features **once** per detector (`yolov8m.pt` for Vynix 1.0, `yolov8x.pt` for Vynix 2.0) so all grid points evaluate in seconds/minutes without re-running CLIP!
   - Tune one parameter at a time (others fixed at nominal defaults):
     - $\sigma \in \{0.04, 0.06, 0.08, 0.10, 0.12, 0.16\}$
     - $\alpha_{\text{override}} \in \{0.15, 0.25, 0.35, 0.45, 0.55\}$ (plus learned $\alpha$)
     - $\beta_{\text{override}} \in \{3.0, 4.5, 5.5, 7.0, 9.0\}$ (plus learned $\beta$)
     - $\lambda \in \{0.7, 0.8, 0.9, 1.0\}$
     - $\gamma_{\text{obj}} \in \{0.05, 0.10, 0.20, 0.50, 1.00\}$
     - $s_{\text{conf}} \in \{0.05, 0.08, 0.12\}$ (requires detector re-filtering on cached proposals if extracted at `0.05`)
3. **Single Frozen Test Evaluation:**
   - Freeze the `val_10`-optimal hyperparameters and evaluate **once** on the `9,658` HICO-DET test images for Vynix 1.0 and Vynix 2.0.
   - Report both **Official 155 Rare / 445 Non-Rare** (and box-level Default IoU $\ge 0.5$ if GT boxes are present in dataset metadata, or clearly state image-level AP if parquet lacks GT boxes) alongside the legacy `h % 4 == 0` split for transparent comparison.

---

## 2. Estimated Runtime & Exact Commands (For User Approval)

| Step | Estimated Time (RTX 3050 Ti) | Command |
|---|---|---|
| 1. Build 90/10 split + extract `train_90` cache & train Vynix 1.0 & 2.0 | ~35–40 min total (~18–20 min each) | `python review_response/run_clean_validation.py --phase train --seed 42` |
| 2. Pre-extract `val_10` (3,812 imgs) features (`s_conf=0.05`) | ~60–70 min total (~30–35 min each) | `python review_response/run_clean_validation.py --phase extract_val --seed 42` |
| 3. Run coordinate-wise grid sweeps on cached `val_10` | ~2–5 min (CPU/GPU tensor math) | `python review_response/run_clean_validation.py --phase tune_val --seed 42` |
| 4. Final frozen test evaluation (ONCE per config) | ~175 min total (~83.5 + 91.7 min) | `python review_response/run_clean_validation.py --phase eval_test --seed 42` |
