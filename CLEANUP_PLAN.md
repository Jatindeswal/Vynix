# Project Vynix: Declutter & Consistency Hardening Plan (`CLEANUP_PLAN.md`)

**Date**: 2026-11-17  
**Branch**: `cleanup` (created from `review-response` at commit `a58607e`)  
**Git Baseline Tag**: `paper-results-v1` (points to `commit a58607e`)  
**Objective**: Hardening repository structure, ensuring single source of truth for evaluation, data splits, and predicates, and eliminating discrepancies between paper text, code, and empirical results.

---

## 1. Safety Rules & Verification Invariants

1. **Strict No-Deletion Policy**: No file is deleted. Obsolete or superseded code/data is moved into `archive/<category>/` with an explicit reason logged in `archive/INDEX.md`.
2. **Untouchable Directories & Artifacts**:
   - Model checkpoints: `saved_models_v1.0/`, `saved_models_v2.0/`, `saved_models_v1.0_train90/`, `saved_models_v2.0_train90/`, `saved_models_vitb16/`, `saved_models_yolov8x_vitb16/`.
   - Feature caches & predictions: `val_caches/`, `test_predictions/`.
   - Audit and benchmark outputs: `review_response/FINAL_TABLES.md`, `review_response/frozen_hyperparameters.json`, `review_response/DISCLOSURES.md`, `review_response/final_test_results_*.json`.
3. **Continuous Regression Testing**:
   - Baseline predictions over a fixed 50-image test set with fixed seeds are saved in `regression_before.pt` and `regression_before.json`.
   - After every single refactoring step, `python run_regression.py --compare regression_before.pt` is executed to assert that predicted triplets (boxes, HOI IDs, and scores) and Box-level Default mAP match identically within $1\times 10^{-6}$.
   - If any regression check fails, the change is reverted immediately.
4. **Anonymity & Portability**:
   - No author names, institutional affiliations, usernames, or absolute host file paths (`D:\...`, `E:\...`, `C:\Users\...`) will appear in code or documentation. Relative paths and environment-variable-backed configurations (`VYNIX_DATASET_DIR`, `VYNIX_CACHE_DIR`) are used exclusively.
5. **Approval Gate**: File moves and directory restructuring will only occur upon user approval of this plan.

---

## 2. Complete File Inventory and Classification

Each file is classified under one of four categories:
- **CORE**: Essential to run inference, train the models, or reproduce reported benchmark numbers and figures.
- **SUPPORT**: Development utilities, package definitions, interactive visualization tools, and test suites.
- **LEGACY**: Superseded scripts (old image-level AP evaluators, $h \bmod 4 == 0$ pseudo-rare code paths, old Colab demos, historical exploratory copies).
- **UNKNOWN**: Artifacts requiring clarification before final archival or retention.

### A. Root Python Scripts & Packages

| File Path | Role / Description | Classification | Proposed Action / Destination | Rationale |
|---|---|:---:|---|---|
| `vynix_fewshot_adapter.py` | Core model architecture, feature extractor, spatial MLP, and soft geometric gate. | **CORE** | Refactor into `src/models/` and `src/pipeline/` | Primary model implementation. Needs module decoupling. |
| `train_vynix_full.py` | Full-dataset cache builder and 15-epoch adapter trainer (`max_shots=50`). | **CORE** | Move to `src/training/train_full.py` | Training script for full-train models. |
| `vynix_cli.py` | Command-line interface for training, cache extraction, and evaluation. | **SUPPORT** | Refactor into `src/cli.py` | Clean CLI dispatcher. |
| `run_regression.py` | Deterministic 50-image regression suite with bit-exact assertion. | **CORE** | Keep in root or `tests/run_regression.py` | Safety harness for refactoring steps. |
| `setup.py` | Python package installation setup. | **SUPPORT** | Retain in root | Standard setup script for editable installs. |
| `evaluate_vynix_full.py` | Historical evaluation script using legacy image-level AP and pseudo-rare splits. | **LEGACY** | Move to `archive/legacy_evaluators/` | Superseded by standard box-level evaluator. |
| `test_vynix_pretrained.py` | Early pre-trained testing script with old prompt templates and image AP. | **LEGACY** | Move to `archive/legacy_evaluators/` | Superseded by unified test evaluator. |
| `vynix_prototype.py` | Earliest 1-stream prototype before 3-stream architecture was introduced. | **LEGACY** | Move to `archive/early_prototypes/` | Historical prototype, inactive. |
| `test_vynix.py` | Fast smoke test generating synthetic tensors. | **SUPPORT** | Move to `tests/test_smoke_dummy.py` | Useful for CI/smoke testing without GPU/dataset. |
| `app_vynix_demo.py` | Streamlit interactive web demonstration app. | **SUPPORT** | Move to `apps/streamlit_app.py` | Interactive presentation utility. |
| `app_vynix_gui.py` | Desktop / browser GUI launcher. | **SUPPORT** | Move to `apps/gui_app.py` | Presentation utility. |
| `generate_paper_figures_v2.py`| Matplotlib script generating Figures 1–5 for the manuscript. | **CORE** | Move to `tools/generate_figures.py` | Reproduces paper figures. |
| `generate_literature_comparison.py`| Generates comparison tables against UPT, SCG, etc. | **SUPPORT** | Move to `tools/generate_literature_table.py` | Literature comparison generator. |
| `analyze_errors.py` | Error analysis script for false-positive detection breakdown. | **SUPPORT** | Move to `tools/analyze_errors.py` | Diagnostic analysis tool. |
| `download_data.py` | Hugging Face parquet downloader for HICO-DET. | **SUPPORT** | Move to `tools/download_data.py` | Data acquisition utility. |
| `download_list_action.py` | Downloader for `list_action.csv`. | **SUPPORT** | Move to `tools/download_list_action.py` | Data acquisition utility. |
| `extract_test_data.py` | Helper script to extract sample test crops. | **SUPPORT** | Move to `tools/extract_test_data.py` | Data debugging tool. |
| `build_ntcc_report.py` | Academic report builder. | **SUPPORT** | Move to `tools/build_report.py` | Report generation tool. |
| `build_colab_presentation.py`| Notebook builder for Colab presentations. | **SUPPORT** | Move to `tools/build_colab.py` | Tooling. |
| `build_html_presentation.py` | HTML presentation builder. | **SUPPORT** | Move to `tools/build_html.py` | Tooling. |
| `build_presentation_deck.py` | PPTX presentation deck builder. | **SUPPORT** | Move to `tools/build_pptx.py` | Tooling. |
| `compile_paper_pdf.py` | Script to render manuscript PDF. | **SUPPORT** | Move to `tools/compile_pdf.py` | Tooling. |
| `generate_word_paper.py` | Script to render manuscript DOCX. | **SUPPORT** | Move to `tools/compile_docx.py` | Tooling. |
| `_verify_report.py` | Verification utility for report formatting. | **SUPPORT** | Move to `tools/_verify_report.py` | Tooling. |

### B. Notebooks

| File Path | Description | Classification | Proposed Action / Destination | Rationale |
|---|---|:---:|---|---|
| `Train_Vynix_Full_Colab.ipynb` | Colab notebook for training Vynix on cloud GPUs. | **SUPPORT** | Move to `notebooks/` | Cloud execution reference. |
| `Vynix_Interactive_Presentation_and_Benchmark.ipynb` | Interactive demonstration and ablation benchmark notebook. | **SUPPORT** | Move to `notebooks/` | Interactive demonstration. |

### C. `review_response/` Directory (Audit & Standard Benchmark)

| File Path | Description | Classification | Proposed Action / Destination | Rationale |
|---|---|:---:|---|---|
| `review_response/evaluate_test_unified.py` | Single-pass test evaluator producing all verified numbers in `FINAL_TABLES.md`. | **CORE** | Move to `src/evaluation/run_test_eval.py` | Official evaluation runner. |
| `review_response/hico_evaluator.py` | Standard HICO-DET Box-Level Default mAP evaluator (`min(IoU_h, IoU_o)>=0.5`). | **CORE** | Move to `src/evaluation/hico_evaluator.py` | Single source of truth for evaluation. |
| `review_response/standard_hico_splits.json` | Official 138 Rare / 462 Non-Rare HOI class IDs. | **CORE** | Move to `splits/standard_hico_splits.json` | Single source of truth for class groups. |
| `review_response/train_90_ids.json` | List of 34,306 image IDs in train-90 partition. | **CORE** | Move to `splits/train_90_ids.json` | Clean validation protocol partition. |
| `review_response/val_10_ids.json` | List of 3,812 image IDs in val-10 partition. | **CORE** | Move to `splits/val_10_ids.json` | Clean validation protocol partition. |
| `review_response/train_train90.py` | Script that trained caches/models on train-90. | **CORE** | Move to `tools/train_train90.py` | Reproducibility script for validation models. |
| `review_response/tune_on_val.py` | Coordinate-wise hyperparameter tuning script on val-10. | **CORE** | Move to `tools/tune_on_val.py` | Reproducibility script for tuning. |
| `review_response/run_extended_sigma_sweep.py`| Extended $\sigma$ sweep script on val-10. | **CORE** | Move to `tools/tune_sigma_extended.py` | Reproducibility script for extended sigma grid. |
| `review_response/audit_vetoes.py` & `audit_vetoes_v1.py`| Veto and rescue hypothesis auditing scripts. | **CORE** | Move to `tools/audit_vetoes.py` | Hypothesis audit reproducibility. |
| `review_response/test_evaluator.py` | Unit test verifying evaluator on toy problem against analytic solution. | **SUPPORT** | Move to `tests/test_hico_evaluator.py` | Regression and unit test. |
| `review_response/evaluate_box_mAP.py` | Predecessor box evaluator script (superseded by `evaluate_test_unified.py`). | **LEGACY** | Move to `archive/legacy_evaluators/` | Superseded by unified runner. |
| `review_response/frozen_hyperparameters.json`| Frozen hyperparameter record. | **CORE** | **RETAIN IN PLACE** | Immutable audit record. |
| `review_response/DISCLOSURES.md` | Comprehensive methodology disclosures. | **CORE** | **RETAIN IN PLACE** | Immutable audit record. |
| `review_response/FINAL_TABLES.md` | Final verified benchmark tables. | **CORE** | **RETAIN IN PLACE** | Immutable benchmark report. |
| `review_response/final_test_results_*.json` | Raw test run evaluation results. | **CORE** | **RETAIN IN PLACE** | Immutable benchmark output. |
| `review_response/00_repo_map.md` to `08_cross_dataset.md`| Technical audit memos for reviewer remarks. | **CORE** | **RETAIN IN PLACE** | Technical review responses. |
| `review_response/REPORT.md`, `RESPONSE_TO_REVIEWERS.md`, `PAPER_PATCHES.md` | Response letters and errata drafts. | **CORE** | **RETAIN IN PLACE** | Reviewer documentation. |

### D. Backup Directories & Artifacts

| File Path | Description | Classification | Proposed Action / Destination | Rationale |
|---|---|:---:|---|---|
| `backup_baseline_3031/*` | Snapshot of codebase before Vynix 2.0 refactoring. | **LEGACY** | Move to `archive/backups/backup_baseline_3031/` | Superseded historical snapshot. |
| `.archify/*` | Intermediate web packaging artifacts. | **LEGACY** | Move to `archive/archify/` | Temporary packaging metadata. |
| `launch_info.json`, `launch_web_gui.bat`, `start_server.vbs` | Launcher scripts for local GUI server. | **SUPPORT** | Move to `apps/launchers/` | Local app support files. |
| `_check_layout.pdf`, `_page*.png`, `_v*.png` | Temporary report render inspection pages. | **LEGACY** | Move to `archive/temp_renders/` | Scratch inspection images. |
| `Vynix_NTCC_Project_Report.docx*`, `Vynix_IEEE_Research_Paper.docx` | Manuscript and report documents. | **SUPPORT** | Move to `docs/` | Documentation files. |
| `list_action.csv` | Official HICO-DET 600 HOI action annotations. | **CORE** | Retain in root or symlink from `data/` | Essential metadata. |

---

## 3. Plan for Single Sources of Truth (Step 2)

### a) Unified Evaluator (`src/evaluation/hico_evaluator.py`)
- Standard Box-Level Default mAP (`min(IoU_h, IoU_o) >= 0.5`) will be the **only** active evaluation metric by default.
- Any legacy image-level AP code will be moved to `legacy_image_level_ap()` with an explicit docstring warning:
  ```python
  """
  WARNING: This function computes unlocalized, image-level multi-label AP.
  It is NOT comparable to published HICO-DET mAP benchmark scores.
  Retained strictly for backward compatibility and ablation audits.
  """
  ```
- All evaluation scripts will import from `src.evaluation.hico_evaluator`.

### b) Canonical Class Groups (`src/data/class_splits.py`)
- Official splits loaded directly from `splits/standard_hico_splits.json`:
  - Standard Rare: 138 classes ($<10$ training instances in HICO-DET).
  - Standard Non-Rare: 462 classes ($\ge 10$ training instances).
- The legacy `h % 4 == 0` pseudo-rare heuristic is deprecated and moved to `archive/`.

### c) Canonical Predicates (`src/data/predicates.py`)
- A single, centralized module defining:
  - `CONTACT_VERBS`: 58 contact predicates.
  - `NON_CONTACT_VERBS`: 59 non-contact predicates (including `no_interaction`).
  - Unit test assertion verifying:
    `len(CONTACT_VERBS) + len(NON_CONTACT_VERBS) == 117` and `len(CONTACT_VERBS.intersection(NON_CONTACT_VERBS)) == 0`.
  - Verbs outside the 117 HICO-DET verb vocabulary are strictly flagged and rejected.

### d) Centralized Configurations (`configs/`)
- `configs/vynix_1.0.yaml` and `configs/vynix_2.0.yaml` will store all hyperparameters:
  - $\sigma$, $\lambda$, $\gamma_{\text{obj}}$, $\tau_{\text{det}}$, top-$K$ pairs ($50$), max shots per class ($50$), random seeds ($42$).
- No magic numbers hard-coded in model or evaluation files.
- `review_response/frozen_hyperparameters.json` remains preserved as the frozen record of tuned values.

### e) Centralized Splits (`splits/`)
- `splits/train_90_ids.json` (34,306 images).
- `splits/val_10_ids.json` (3,812 images).
- `splits/standard_hico_splits.json` (138 Rare / 462 Non-Rare).
- `splits/CHECKSUMS.sha256`: SHA-256 integrity verification file for all split ID lists.

### f) Environment-Aware Paths (`configs/paths.yaml`)
- Paths configured via environment variables with sensible defaults:
  - `VYNIX_DATASET_DIR`: defaults to `data/` or relative path.
  - `VYNIX_MODELS_DIR`: defaults to `saved_models/`.
  - `VYNIX_PREDICTIONS_DIR`: defaults to `test_predictions/`.
- Elimination of hard-coded Windows drive letters (`D:`, `E:`) across the codebase.

---

## 4. Latent Bug Fixes & Refactoring Plan (Step 3)

1. **Pass `conf` Explicitly to YOLO**:
   - In `MultiStreamFeatureExtractor.detect(pil_img, conf=None)`:
     Pass `conf = self.detector_conf if conf is None else conf` to `self.yolo(pil_img, conf=conf, device=self.device, verbose=False)`.
   - Maintain a `legacy_yolo_conf=True` compatibility mode (defaulting to $0.25$) so existing checkpoints and reported numbers replicate identically.
   - Add a unit test verifying that setting `conf=0.08` produces more candidates than `conf=0.25`.
2. **Standardize `max_shots` Everywhere**:
   - Rename all instances of `max_shots` to `max_shots_per_class = 50`.
   - Ensure consistency across docstrings, CLI arguments, and documentation.
3. **Harmonize Geometric Veto Documentation**:
   - State clearly in code docstrings and comments:
     - Implementation formula: $\exp(-(d_{\text{box}} / \sigma)^2)$.
     - Theoretical manuscript notation: $\exp(-d_{\text{box}}^2 / (2\sigma^2))$ (equivalent to scaling $\sigma_{\text{code}} = \sigma_{\text{paper}} \sqrt{2}$).
     - Do not modify computation; document exact algebraic equivalence.
4. **Dead Code Removal**:
   - Strip unused imports, commented-out debugging blocks, and redundant helper copies.

---

## 5. Test Suite Plan (`tests/`)

Single command: `python -m pytest tests -q`
1. `test_evaluator_toy.py`: Evaluator reproduces known analytic AP on synthetic test cases.
2. `test_class_splits.py`: Assertions on 138 Rare / 462 Non-Rare class IDs and disjointness.
3. `test_predicates.py`: Verify 58 contact / 59 non-contact verbs ($58+59=117$).
4. `test_configs.py`: Verify YAML configs match `frozen_hyperparameters.json`.
5. `test_split_integrity.py`: Verify SHA-256 checksums of `train_90_ids.json` and `val_10_ids.json`.
6. `test_detector_threshold.py`: Assert candidate pair count changes when `conf` is altered.
7. `test_regression.py`: Assert 50-image predictions match `regression_before.pt` within $1\times 10^{-6}$.

---

## 6. Traceability & Manifests (Steps 5 & 6)

- `PAPER_TO_CODE.md`: Table mapping every paper equation, hyperparameter, table, and figure to source code files and commit hashes, including the "What the paper says vs what the code does" section.
- `README.md`: Environment setup, reproducible commands, and dataset instructions.
- `Makefile` / `run_all.md`: Ordered sequence of reproduction commands.
- `results/MANIFEST.csv`: Index of all generated results and data artifacts.

---

*Awaiting user approval before moving files or restructuring directories.*
