# Task 6 — Validating the Geometric Veto (Answers Reviewer A)

**Status:** Existing full-test ablation numbers are recorded `[LOG]`. Granular $\sigma$-sensitivity on the clean `val_10` split, contact-vs-non-contact mAP breakdown, and hypothesis-level veto precision/recall curves are `[UNKNOWN]` (blocked by `run_command` CWD error and awaiting Task 3 `val_10` execution approval).

---

## 1. Existing Test-Set Gate Evidence (`[LOG]`)

From `full_test_ablation_results.json` (`run_full_test_ablation.py`, 9,658 test images, Vynix 2.0 `YOLOv8x + ViT-B/16`):
- **Soft Gate ($\sigma=0.08$):** `34.80%` Full / `36.10%` Rare / `34.37%` Non-Rare (`272,946` vetoes) `[LOG]`
- **w/o Geometric Gate ($\gamma=1$):** `34.53%` Full / `36.04%` Rare / `34.02%` Non-Rare (`0` vetoes) -> Soft gate gives `+0.28%` Full mAP (`+0.35%` Non-Rare) `[LOG]`
- **Hard Binary Gate ($\text{IoU}>0$):** `34.33%` Full / `35.62%` Rare / `33.90%` Non-Rare (`463,346` vetoes) -> Hard gate *hurts* by `-0.20%` vs no gate (`-0.48%` vs soft gate) `[LOG]`

From `analysis_outputs/vynix_comprehensive_ablation_results.json` (Phase 6 `run_ablation_studies.py` hardcoded data):
- Reports a larger gain for Vynix 1.0 (`26.15%` no gate -> `28.23%` hard -> `30.31%` soft $\sigma=0.08$), with $\sigma \in \{0.02, 0.04, 0.08, 0.12, 0.20\}$ `[LOG]`. **Caution:** These Phase 6 ablation numbers are hardcoded in `run_ablation_studies.py` without an underlying raw sweep script in the repo `[LOG]`.

---

## 2. Planned Clean Validation Measurements (Once Approved & Unblocked)

1. **$\sigma$ Sensitivity on `val_10` (Task 3):** Evaluate $\sigma \in \{0.04, 0.06, 0.08, 0.10, 0.12, 0.16\}$ (and compare `exp(-(d/sigma)^2)` vs `exp(-d^2/(2*sigma^2))`).
2. **Contact vs. Non-Contact mAP Breakdown:** Aggregate per-class AP separately over the `59` gated contact-verb HOIs vs. the remaining non-contact HOIs with and without the gate (and on `04_veto_audit.py` output).
3. **Hypothesis-Level Precision/Recall:** Using `audit_vetoes.py`, report TP-lost rate vs. FP-suppression rate at `gate < 0.05` and across thresholds.
