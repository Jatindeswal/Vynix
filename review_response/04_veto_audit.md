# Task 4 — Veto & Rescue Count Audit (Answers Reviewer B.3)

**Audit Status:** Exact code definitions and internal arithmetic contradictions established `[LOG]`. Full 9,658-image per-hypothesis recomputation (split by GT-matched TP vs unmatched FP) is `[UNKNOWN]` until `review_response/audit_vetoes.py` is executed (blocked by `run_command` CWD issue).

---

## 1. Exact Definitions in Code (`[LOG]`)

1. **Candidate `(pair, verb)` Hypotheses:**
   - In `test_vynix_pretrained.py` and `run_full_test_ablation.py:205-234`, for each image:
     - YOLO (`detector_conf=0.08`) detects persons and objects.
     - All `(person, object)` pairs are formed and truncated to top-50 by `p_conf * o_conf` (`pairs[:50]`).
     - Pairs whose object class does not map to a HICO-DET object (`meta.obj_to_entries`) are skipped.
     - Each remaining pair is expanded over `entries` (all HICO-DET verbs valid for that object, including `no_interaction`), forming `(pair, verb, hoi_id)` hypotheses.
2. **Exact Code Definition of "Vetoed" (`[LOG]`):**
   - `test_vynix_pretrained.py:144` & `run_full_test_ablation.py:291`:
     ```python
     if gate_val < 0.05:
         veto_counts[s_id] += 1
     ```
   - Note that the soft gate `compute_soft_geometric_gate` (`vynix_fewshot_adapter.py:187`) actually computes $\exp(-(d_{\text{box}}/\sigma)^2)$ (missing the `2` in $2\sigma^2$). Thus `gate_val < 0.05` triggers when $v \in \mathcal{V}_{\text{contact}}$, $\text{IoU} = 0$, and $d_{\text{box}} > \sigma \sqrt{\ln 20} \approx 1.7308\,\sigma \approx 0.1385$ (for $\sigma = 0.08$).
   - Under the **hard binary gate** (`run_full_test_ablation.py:276`), `hard_gate = 0.0 if (verb in CONTACT_VERBS and iou_val == 0.0) else 1.0`, so `gate_val < 0.05` counts every contact-verb hypothesis with $\text{IoU} = 0$.
3. **Exact Definition of "Rescued" (`[LOG]`):**
   - There is **NO** `rescued` counter in `test_vynix_pretrained.py` or `run_full_test_ablation.py` `[LOG]`.
   - In `process.md:149, 200-206` and `analysis_outputs/vynix_comprehensive_ablation_results.json:15,51-52`, "Rescued Contacts" (`54,532` for $\sigma=0.08$) represents the reduction in vetoes relative to the Hard Binary Gate (`IoU == 0`): i.e., contact-verb hypotheses with $\text{IoU} = 0$ whose soft gate is $\ge 0.05$ ($d_{\text{box}} \le 0.1385$ under the code's formula).

---

## 2. Comparison Against Paper Numbers (`[LOG]` & Arithmetic Checks)

| Metric | Paper Claim | Repo Log Value | Source | Status / Discrepancy |
|---|---|---|---|---|
| **Vynix 1.0 Soft Vetoes** (`yolov8m`, $\sigma=0.08$) | `265,189` | `265,189` | `saved_models_v1.0/eval_results.json:5`, `saved_models/eval_results.json:5` | Matches repo log `[LOG]`; independent recomputation `[UNKNOWN]` |
| **Vynix 1.0 Hard Vetoes** (`yolov8m`, $\text{IoU}=0$) | `319,803` (implied by rescue) | `319,803` | `analysis_outputs/vynix_comprehensive_ablation_results.json:15`, `process.md:200` | **Arithmetic discrepancy:** `319,803 - 265,189 = 54,614`, NOT `54,532` (82-hypothesis gap!) `[LOG]` |
| **Vynix 1.0 Rescued** | `54,532` | `54,532` | `analysis_outputs/vynix_comprehensive_ablation_results.json:52`, `process.md:149,204` | Contradicts `319,803 - 265,189 = 54,614` by **82**; note that for all other $\sigma$ rows in `vynix_comprehensive_ablation_results.json` (`0.02, 0.04, 0.12, 0.20`), `vetoes + rescued == 319,803` exactly, whereas at $\sigma=0.08$, `265,189 + 54,532 = 319,721` `[LOG]` |
| **Vynix 2.0 Soft Vetoes** (`yolov8x`, $\sigma=0.08$) | `272,946` | `272,946` | `saved_models_v2.0/eval_results.json`, `full_test_ablation_results.json:13` | Matches repo log `[LOG]`; independent recomputation `[UNKNOWN]` |
| **Vynix 2.0 Hard Vetoes** (`yolov8x`, $\text{IoU}=0$) | `463,346` | `463,346` | `full_test_ablation_results.json:33` | Matches repo log `[LOG]`; independent recomputation `[UNKNOWN]` |
| **Vynix 2.0 Extra Hard Vetoes** | `190,400` | `463,346 - 272,946 = 190,400` | `full_test_ablation_results.json:13,33` | Exact arithmetic match (`463,346 - 272,946 = 190,400`) `[LOG]` |

---

## 3. Standalone Verification Script

`review_response/audit_vetoes.py` is provided to recompute from scratch on all `9,658` test images (for both `yolov8m.pt` and `yolov8x.pt`):
- Total candidate `(pair, verb)` hypotheses;
- Soft gate vetoes (`gate < 0.05` under both code formula $\exp(-(d/\sigma)^2)$ and paper formula $\exp(-d^2/(2\sigma^2))$);
- Hard gate vetoes (`IoU == 0` for `verb in CONTACT_VERBS`);
- Rescued hypotheses (`IoU == 0` and `gate >= 0.05`);
- Split by **GT-matched** (image-level `hoi_id in pos_set`, and box-level if available) vs **unmatched** (false positives suppressed).
Output CSV: `review_response/04_veto_audit_results.csv` (`[UNKNOWN]` until executed).
