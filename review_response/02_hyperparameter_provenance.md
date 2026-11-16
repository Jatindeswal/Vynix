# Task 2 — Hyperparameter Provenance (Answers Reviewer B.2 and B.5)

**Audit Verdict:** Every tuned inference hyperparameter in Vynix 2.0 ($\lambda=0.90$, $\gamma_{\text{obj}}=0.10$, prompt-ensemble toggle, adaptive-$\alpha$ toggle) and the Phase 6/8 ablations were selected or validated **using the HICO-DET test set** (either the first 300 test images via `test_vynix_pretrained.py --limit 300`, or full 9,658 test images). **No train-derived validation split exists anywhere in the repository.**

---

## 1. Hyperparameter Provenance Table

| Parameter | Claimed / Active Value | Where Set (`file:line`) | Selected Using Which Split | Evidence | Verdict |
|---|---|---|---|---|---|
| $\sigma$ (soft gate bandwidth) | `0.08` (Note: `process.md:349` says `0.15` for Phase 8, but code uses `0.08`) | `vynix_fewshot_adapter.py:175` (`sigma_contact: float = 0.08`) | **test-involved** / **unknown** | `process.md:197-206` & `run_ablation_studies.py` sweep $\sigma \in \{0.02, 0.04, 0.08, 0.12, 0.20\}$ on test mAP; initial choice of `0.08` has no train-val log | `test-involved` |
| $\alpha$ (cache residual weight) | Paper claims fixed `0.35`; code uses **learned `nn.Parameter` init `1.0`** | `vynix_fewshot_adapter.py:524` (`nn.Parameter(torch.tensor(1.0))`); `process.md:225-232` | **test-involved** (for paper's `0.35` sweep) / **train** (for AdamW) | `run_ablation_studies.py` sweeps $\alpha \in [0,1]$ on test mAP; `process.md:332` states overriding learned $\alpha$ hurt 300-img test-slice mAP by `-0.52%`, so learned $\alpha$ was kept | `test-involved` |
| $\beta$ (affinity sharpness) | `5.5` (learned `nn.Parameter` init `5.5`) | `vynix_fewshot_adapter.py:525` (`nn.Parameter(torch.tensor(5.5))`) | **test-involved** / **unknown** | Tip-Adapter default / swept on test mAP in `process.md:234-240` (`1, 3, 5.5, 8, 12`) | `test-involved` |
| $\lambda$ (detector score exponent) | `1.0` (v1.0) / `0.90` (v2.0) | `run_option1_retrain_and_eval.py:59,75` (`--det-lambda 0.9`); `test_vynix_pretrained.py:195` (CLI default `0.8`) | **test-involved** | `process.md:313, 318-329, 335` explicitly tuned $\lambda \in \{0.6, 0.8, 0.9, 1.0\}$ on the **300-image test slice** (`--limit 300`) | `test-involved` |
| $\gamma_{\text{obj}}$ (object semantic gate) | `0.10` (v2.0) | `run_option1_retrain_and_eval.py:59,75` (`--obj-gamma 0.1`); `test_vynix_pretrained.py:198` (CLI default `0.15`) | **test-involved** | `process.md:318-329, 334` compared $\gamma \in \{0.0, 0.10, 0.15\}$ on the **300-image test slice** (`--limit 300`) | `test-involved` |
| Detector score threshold $s_{\text{conf}}$ | `0.08` | `train_vynix_full.py:262`, `test_vynix_pretrained.py`, `run_full_test_ablation.py:111` | **test-involved** / **unknown** | `process.md:242-249` sweeps $\{0.03, 0.05, 0.08, 0.15, 0.25\}$ against test mAP | `test-involved` |
| Top-$K$ pairs per image | `50` (test), `20` (train cache) | `vynix_fewshot_adapter.py:798-800`, `run_full_test_ablation.py:206-208`, `train_vynix_full.py:135` | **unknown** | Hardcoded in scripts; no validation sweep logged | `unknown` |
| `max_shots_per_class` | `50` | `train_vynix_full.py:247` | **unknown** | Engineering cap added to reduce extraction time from ~5h to 17.33 min (`process.md:136`) | `unknown` |
| MLP epochs | `15` | `train_vynix_full.py:243`, `run_full_benchmark_and_train_vitb16.py:48`, `run_option1_retrain_and_eval.py:47` | **unknown** | Hardcoded `15` (function default is `20` at `train_vynix_full.py:203`) | `unknown` |
| Optimizer / LR / Weight Decay | `AdamW`, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=64` (v2.0) / `256` (v1.0) | `train_vynix_full.py:203, 207, 244`; `run_option1_retrain_and_eval.py:48`; `run_full_benchmark_and_train_vitb16.py:49` | **unknown** | Standard defaults; no LR/optimizer sweep logged | `unknown` |
| Spatial log-prior weight | `0.5` | `vynix_fewshot_adapter.py:562` (`+ 0.5 * spatial_log_prior`) | **unknown** | Hardcoded; not reported in the paper | `unknown` |
| Veto reporting threshold | `gate < 0.05` | `test_vynix_pretrained.py:144`, `run_full_test_ablation.py:291` | **unknown** | Hardcoded logging threshold (`gate < 0.05`) | `unknown` |

---

## 2. Complete Inventory of Test-Split Evaluation Runs Found in Logs/Artifacts

1. **Zero-Shot Prototype (Hard Gate):** `22.17%` Full / `18.57%` Rare / `23.42%` Non-Rare, `319,803` vetoes (`process.md:78, 118, 185`).
2. **Milestone 1 (`YOLOv8n + ViT-B/32`, Hard Gate):** `22.03% / 23.32% / 21.59%`, `319,803` vetoes (`process.md:93, 140-144`).
3. **Milestone 2 (`YOLOv8m + ViT-B/32`, Soft Gate):** `28.23% / 28.98% / 27.98%`, `265,189` vetoes (`saved_models/eval_results.json`, `process.md:146-150`).
4. **Milestone 3 / Vynix 1.0 (`YOLOv8m + ViT-B/16`, Soft Gate):** `30.31% / 29.71% / 30.51%`, `265,189` vetoes (`saved_models_v1.0/eval_results.json`, `process.md:152-163`).
5. **Phase 8 300-Image Test-Slice Tuning (6 runs on `--limit 300`):** Swept prompt ensembling, adaptive $\alpha$, $\gamma_{\text{obj}} \in \{0.0, 0.10, 0.15\}$, `YOLOv8m` vs `YOLOv8x`, and $\lambda \in \{0.90, 1.00\}$ (`process.md:318-329`). **Parameters changed across all 6 runs while observing test-slice mAP.**
6. **Phase 8 Intermediate (`YOLOv8x` + unaligned v1.0 cache, $\lambda=0.9, \gamma=0.1$):** `31.71% / 30.63% / 32.07%`, `272,946` vetoes (`saved_models_vitb16_backup_3171/eval_results.json`, `process.md:343-364`).
7. **Vynix 2.0 300-Image Test-Slice Check:** `run_option1_retrain_and_eval.py:52-61` ran `test_vynix_pretrained.py --limit 300` before full test eval `[LOG]`.
8. **Vynix 2.0 Flagship (`YOLOv8x` aligned cache, $\lambda=0.9, \gamma=0.1$):** `34.80% / 36.10% / 34.37%`, `272,946` vetoes (`saved_models_v2.0/eval_results.json`).
9. **Vynix 2.0 6-Stream Full Test Ablation:** `run_full_test_ablation.py` evaluated 6 variants simultaneously on all 9,658 test images (`full_test_ablation_results.json`).
- **Total distinct test-set evaluations documented:** **9 runs/suites** (14 individual configurations if counting the 6-run Phase 8 300-img slice individually; 19 if counting the 6 streams in `full_test_ablation_results.json`).

**Bottom Line for Reviewer B.2 / B.5:** The authors **did** tune $\lambda$, $\gamma_{\text{obj}}$, and adaptive-$\alpha$ on a 300-image slice of the **test set** (`--limit 300`), and used test-set ablations for $\sigma, \alpha, \beta, s_{\text{conf}}$. A clean train-derived validation protocol (Task 3) is **mandatory** before submitting the rebuttal.
