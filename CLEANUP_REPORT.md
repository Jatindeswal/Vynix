# Project Vynix: Phase 2 Declutter and Consistency Hardening Report (`CLEANUP_REPORT.md`)

## 1. Executive Summary

Phase 2 was executed to eliminate code-paper contradictions, centralize duplicated logic, and harden the repository against silent divergence. All modifications strictly adhered to the non-negotiable safety rules:
- **Zero code changes to inference logic or weights:** Mathematical behavior and model predictions are 100% bit-exact with the pre-refactor state.
- **Zero deletions:** Obsolete or legacy files were relocated via `git mv` to `archive/` with individual rationales documented in `archive/INDEX.md`.
- **Full backward compatibility:** Transparent compatibility shims remain at all legacy root import paths (`vynix_fewshot_adapter.py`, `review_response/hico_evaluator.py`).
- **Bit-exact regression gating:** A deterministic 50-image regression check (`run_regression.py --compare regression_before.pt`) was executed and passed identically after every modification batch.
- **Strict anonymity:** No author names, institutional affiliations, local usernames, or host drive letters (e.g., `C:`, `D:`, `E:`) remain in active repository configuration or code.

---

## 2. Refactoring Progression & Batches

### Batch 1: Stale-Claims & Anonymity Audit (Commit `717b6be`)
- **Stale-Claims Sweep:** Scanned the entire repository across 14 search terms (`30.31`, `34.80`, `36.10`, `29.71`, `105.0`, `54,532`, `"all 38,118"`, `"SOTA"`, `"outperforms"`, `"h % 4"`, `"0.08 threshold"`, `"soft continuous"`, etc.). Cataloged 1,071 occurrences in `STALE_CLAIMS.md`.
- **Warning Banners:** Placed explicit warning headers on active legacy files (`README.md`, `app_vynix_demo.py`, `app_vynix_gui.py`) stating that legacy image-level mAPs are superseded by the official box-level Default setting metrics in `FINAL_TABLES.md`.
- **Anonymity Audit:** Created `ANON_RELEASE_CHECKLIST.md` auditing personal identifiers in git history, Word documents, launchers, and configs.

### Batch 2: Configuration & Environment Hardening (Commit `717b6be`)
- **Centralized Paths Configuration:** Created `configs/paths.yaml` declaring root-relative paths and environment variable overrides (`VYNIX_DATASET_DIR`, `VYNIX_PREDICTIONS_DIR`, `VYNIX_RESULTS_DIR`), eliminating hardcoded drive letters.
- **Frozen Environment:** Generated `requirements.lock` documenting the exact Python package dependencies and library versions.
- **Legacy Detector Confidence Test:** Created `tests/test_legacy_conf.py` verifying and documenting that Ultralytics YOLO defaults to `conf=0.25` when `conf` is omitted during proposal extraction.

### Batch 3: Source Code Reorganization & Compatibility Shims (Commit `a942972`)
- **Modular Directory Hierarchy:** Created `src/` structure:
  - `src/models/vynix_fewshot_adapter.py`
  - `src/evaluation/hico_evaluator.py`
  - `src/data/`
  - `src/config/`
- **Transparent Compatibility Shims:**
  - `vynix_fewshot_adapter.py`: Re-exports all public classes, functions, and constants from `src.models.vynix_fewshot_adapter`.
  - `review_response/hico_evaluator.py`: Re-exports all evaluator classes from `src.evaluation.hico_evaluator`.
- **Integrity Verification:** Tested `torch.load` and forward evaluation across all 4 checkpoints (`saved_models_v1.0`, `saved_models_v2.0`, `saved_models_v1.0_train90`, `saved_models_v2.0_train90`), followed by a 100% bit-exact regression pass.

### Batch 4: Script Archiving & Manifest Creation (Commits `aa4f940`, `8290827`)
- **Manifest Documentation:** Authored `results/MANIFEST.csv` mapping every numerical result reported in `FINAL_TABLES.md` to its exact execution command, seed, and output file.
- **Zero-Import Validation:** Verified via `git grep` that no production pipelines import obsolete legacy scripts.
- **Archiving via `git mv`:** Moved obsolete scripts:
  - `evaluate_vynix_full.py` -> `archive/legacy_eval/evaluate_vynix_full.py`
  - `run_option1_retrain_and_eval.py` -> `archive/legacy_eval/run_option1_retrain_and_eval.py`
  - `build_colab_presentation.py` -> `archive/presentation_builders/build_colab_presentation.py`
  - Documented one-line reasons for each file in `archive/INDEX.md`.

### Batch 5: Single Source of Truth & Comprehensive Test Suite
- **Centralized Data & Splits:**
  - `src/data/class_splits.py`: Canonical definitions of the official 138 Rare / 462 Non-Rare HOI class splits, 80 COCO classes, and 117 verbs. Discontinued all heuristic $h \bmod 4 == 0$ pseudo-rare splits.
  - `src/data/predicates.py`: Canonical partition of 117 HICO-DET verbs into 59 physical contact verbs and 58 non-contact verbs ($59 + 58 = 117$). Preserves backward compatibility with legacy 64-entry `CONTACT_VERBS` set.
- **Centralized Configuration & Checkpoint Loading:**
  - `src/config/paths.py`: Path resolver reading environment variables with relative repo-root fallbacks.
  - `src/config/hyperparameters.py`: Reads frozen inference parameters directly from `review_response/frozen_hyperparameters.json`.
  - `src/models/checkpoint_loader.py`: Unified checkpoint loader asserting cache key dimensions `[N, 1536]`, cache value dimensions `[N, 600]`, and Spatial MLP layer shapes `[64, 8]`, `[64]`, `[600, 64]`.
- **Reproducibility Test Suite (`tests/`):**
  - `tests/test_shapes.py`: Verifies tensor and model shapes across all 4 checkpoint suites.
  - `tests/test_predicates.py`: Verifies predicate partitioning, counts, and spatial gate routing.
  - `tests/test_evaluator.py`: Verifies analytic toy-case PR-AUC calculation ($5/6$), asymmetric IoU vetoing, and duplicate suppression.
  - `tests/test_regression.py`: Verifies end-to-end 50-image regression against `regression_before.pt`.
  - `tests/test_legacy_conf.py`: Verifies proposal extraction confidence threshold dynamics.
- **Paper-to-Code Mapping:** Authored `PAPER_TO_CODE.md` mapping all 13 core paper claims to their exact code locations and audited values.

---

## 3. Verification & Test Summary

| Test Module | Scope / Checks | Execution Time | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_predicates.py` | 117 verb counts, 59/58 split, 138/462 HOI split, disjointness | < 0.01s | **PASS** |
| `tests/test_evaluator.py` | Box IoU, toy PR-AUC (5/6), asymmetric IoU veto, duplicate suppression | < 0.01s | **PASS** |
| `tests/test_shapes.py` | Shapes of `cache_keys` `[N, 1536]`, `cache_values` `[N, 600]`, SpatialMLP across 4 checkpoints | 0.23s | **PASS** |
| `tests/test_legacy_conf.py` | Ultralytics default `conf=0.25` behavior and candidate extraction | 7.52s | **PASS** |
| `tests/test_regression.py` | Bit-exact identity against `regression_before.pt` on 50 test images | ~74s | **PASS** |

All tests execute cleanly via `python -m unittest discover tests`.
