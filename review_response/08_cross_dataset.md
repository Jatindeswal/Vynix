# Task 8 — Cross-Dataset Validation (V-COCO) Plan (Answers Reviewer A)

**Status:** `[UNKNOWN]` / Not executable in this session. V-COCO is **not** present in the repository or `dataset/` (`E:\Dataset` contains HICO-DET parquet files and `list_action.csv` only `[LOG]`).

---

## 1. Feasibility & Exact Requirements for V-COCO

1. **Dataset Download (~2–3 GB + COCO val2014):**
   - Requires V-COCO annotations (`vcoco_train.json`, `vcoco_val.json`, `vcoco_test.json`, 26 action classes, ~10,346 images from COCO `train2014`/`val2014`) and `vsrl_eval.py`.
2. **Code Changes Needed (Isolated in `review_response/`):**
   - Replace `HOIMeta(dataset_dir)` (`list_action.csv`) with a `VCOCOMeta` mapping COCO's 80 object classes + 26 V-COCO roles/actions (including `agent` vs `instr`/`obj` role distinction).
   - Map V-COCO's 26 action verbs to `CONTACT_VERBS` (e.g., `hold`, `ride`, `sit`, `eat`, `drink`, `cut`, `hit`, `kick`, `carry`, `read`, `look`, `talk_on_phone`, `work_on_computer`, etc.).
   - Implement official V-COCO `eval_vcoco` (`AP_role` Scenario 1 & Scenario 2, box $\text{IoU} \ge 0.5$).
3. **Estimated Effort & Runtime (RTX 3050 Ti, 4 GB VRAM):**
   - **Engineering:** ~2–4 hours to write `VCOCOMeta` + official `vsrl_eval` adapter.
   - **Compute:** Train/val cache extraction on V-COCO `trainval` (~5,400 images) ~10–15 min; `test` (~4,946 images) ~40–45 min.
4. **Recommendation for Rebuttal:**
   - If time/compute permits after fixing HICO-DET box-level evaluation and Task 3 clean validation, run V-COCO `AP_role` Scenario 1. Otherwise, acknowledge V-COCO as future work and focus rebuttal compute on clean HICO-DET validation + box-level Default mAP.
