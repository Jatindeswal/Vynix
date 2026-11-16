# Phase 0 — Read-Only Repository & Pipeline Map

**Audit Status:** Static inspection `[LOG]` complete. Shell/git/Python execution (`run_command`) is currently `[UNKNOWN]` / blocked because the execution harness attempts to start in a non-existent drive path (`e:\...` instead of the actual workspace drive). Consequently, `git checkout -b review-response` and `git log -p` could not be executed in this session.

---

## 1. Entry Points (Training, Evaluation, Inference)

| Role | File | Key Function / CLI Entry | Source Reference |
|---|---|---|---|
| **Training (Full Dataset)** | `train_vynix_full.py` | `main()`, `build_full_dataset_cache()`, `train_adapter_full()` | `[LOG]` `train_vynix_full.py:24-315` |
| **Evaluation (Single Model)** | `test_vynix_pretrained.py` | `main()`, `evaluate_pretrained()` | `[LOG]` `test_vynix_pretrained.py:28-303` |
| **Evaluation (6-Stream Ablation)** | `run_full_test_ablation.py` | `main()`, `evaluate_stream()` | `[LOG]` `run_full_test_ablation.py:57-424` |
| **Orchestrator (Vynix 1.0)** | `run_full_benchmark_and_train_vitb16.py` | Runs `test_vynix_pretrained.py` (B/32) -> `train_vynix_full.py` (YOLOv8m + ViT-B/16) -> `test_vynix_pretrained.py` | `[LOG]` `run_full_benchmark_and_train_vitb16.py:25-64` |
| **Orchestrator (Vynix 2.0)** | `run_option1_retrain_and_eval.py` | Runs `train_vynix_full.py` (YOLOv8x + ViT-B/16) -> 300-img test slice -> full test eval | `[LOG]` `run_option1_retrain_and_eval.py:1-82` |
| **Core Model & Helpers** | `vynix_fewshot_adapter.py` | `HOIMeta`, `SpatialMLP`, `MultiStreamFeatureExtractor`, `Vynix3StreamAdapter`, `compute_soft_geometric_gate` | `[LOG]` `vynix_fewshot_adapter.py:224-900` |
| **CLI Wrapper** | `vynix_cli.py` | `predict`, `eval`, `train` subcommands | `[LOG]` `vynix_cli.py:1-250` |
| **Phase 6 Plotter (Hardcoded)** | `run_ablation_studies.py` | Hardcoded Phase 6 ablation tables + chart generation | `[LOG]` `run_ablation_studies.py:1-223` |

---

## 2. Core Mechanism Locations

| Component | Paper Claim | Actual Implementation Location & Behaviour | Tag |
|---|---|---|---|
| **Geometric Veto Gate $\gamma$** | $\exp(-d_{\text{box}}^2 / (2\sigma^2))$, $\sigma=0.08$ | `vynix_fewshot_adapter.py:175-188`: `math.exp(-(b_dist / max(sigma_contact, 1e-4))**2)` (**missing factor of 2 in denominator**) | `[LOG]` |
| **Contact Verb Set $\mathcal{V}_{\text{contact}}$** | 64 physical contact verbs | `vynix_fewshot_adapter.py:79-88`: hardcoded set `CONTACT_VERBS` (64 verbs) | `[LOG]` |
| **Object Semantic Gate $G_{\text{obj}}$** | $(\max(0, f_o \cdot t_{\text{obj}}))^{\gamma_{\text{obj}}}$, $\gamma_{\text{obj}}=0.10$ | `test_vynix_pretrained.py` & `run_full_test_ablation.py:264-282`; weights via `build_object_text_weights` (`vynix_fewshot_adapter.py:475-500`) | `[LOG]` |
| **Detector Score Calibration $\lambda$** | $(s_h \cdot s_o)^\lambda$, $\lambda=0.90$ (v2.0) / $1.0$ (v1.0) | `test_vynix_pretrained.py:129`, `run_full_test_ablation.py:281` | `[LOG]` |
| **Pairwise Cache & Blending** | $k_e = [f_v; f_{\text{geom}}] \in \mathbb{R}^{1792}$; fixed $\alpha=0.35, \beta=5.5$; $(1-\alpha)S_{\text{vlm}} + \alpha S_{\text{cache}}$ | `vynix_fewshot_adapter.py:507-564`: `cache_keys` is $\mathbb{R}^{1536}$ (visual only); `alpha` (init `1.0`) and `beta` (init `5.5`) are `nn.Parameter`s; `clip_logits + alpha * cache_logits + 0.5 * log(spatial_priors + 1e-6)` | `[LOG]` |
| **Spatial MLP** | `Linear(8,128)-GELU-Linear(128,256)-GELU` | `vynix_fewshot_adapter.py:249-266`: `Linear(8, 64)-LayerNorm(64)-ReLU-Linear(64, 600)-Sigmoid` | `[LOG]` |
| **Calibration Matrix $M_{c_o, a}$** | Explicit matrix $M_{c_o,a}$ in $S_{\text{HOI}}$ | Implemented implicitly as object-filtered verb lookup `meta.obj_to_entries[hico_name]` (`vynix_fewshot_adapter.py:242, 808`) | `[LOG]` |
| **Feature Caching** | Training-time feature caching | `train_vynix_full.py:24-201`: shard-level disk cache `cache_shard_XX.pt` + merged `vynix_full_cache.pt` with `max_shots_per_class=50` | `[LOG]` |
| **Rare / Non-Rare Split** | 155 Rare ($<10$ train instances) / 445 Non-Rare | `test_vynix_pretrained.py:173`, `vynix_fewshot_adapter.py:847`, `run_full_test_ablation.py:66`: `rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}` (**150 pseudo-rare classes, NOT 155**) | `[LOG]` |
| **Evaluation Metric** | HICO-DET Default setting (both boxes IoU $\ge 0.5$) | `test_vynix_pretrained.py:165-170`, `vynix_fewshot_adapter.py:839-844`, `run_full_test_ablation.py:57-64`: **Image-level multi-label AP** (`all_preds[img_idx][hoi_id] = max_pair conf`, `all_gt[img_idx] = set(positive_objects)`), no box IoU matching | `[LOG]` |

---

## 3. Existing Logs & Test-Set Evaluation Runs Found

| Run / Artifact | Path | Detector / CLIP | Full / Rare / Non-Rare mAP | Vetoes | Images | Tag |
|---|---|---|---|---|---|---|
| Milestone 2 (B/32) | `saved_models/eval_results.json` | `yolov8m.pt` / `ViT-B/32` | `28.23 / 28.98 / 27.98` | `265,189` | `[UNKNOWN]` (not in JSON; `process.md:143` says 9,658) | `[LOG]` |
| Vynix 1.0 Flagship | `saved_models_v1.0/eval_results.json` | `yolov8m.pt` / `ViT-B/16` | `30.31 / 29.71 / 30.51` | `265,189` | `9,658` | `[LOG]` |
| Phase 8 Intermediate | `saved_models_vitb16_backup_3171/eval_results.json` | `yolov8x.pt` / `ViT-B/16` ($\lambda=0.9, \gamma=0.1$) | `31.71 / 30.63 / 32.07` | `272,946` | `9,658` | `[LOG]` |
| Phase 8 300-img Slice | `process.md:318-329` (6 configs) | `yolov8m.pt` & `yolov8x.pt` | `22.81` to `24.16` | `6,841` / `7,357` | `300` (test slice) | `[LOG]` |
| Vynix 2.0 Flagship | `saved_models_v2.0/eval_results.json` | `yolov8x.pt` / `ViT-B/16` ($\lambda=0.9, \gamma=0.1$) | `34.80 / 36.10 / 34.37` | `272,946` | `9,658` | `[LOG]` |
| Vynix 2.0 6-Stream Ablation | `run_full_test_ablation.py` + `full_test_ablation_results.json` | `yolov8x.pt` / `ViT-B/16` | `34.80` (base), `34.53`, `34.33`, `34.83`, `34.77`, `34.51` | `272,946` / `463,346` / `0` | `9,658` | `[LOG]` |
| Engineering Log | `process.md` | Chronicles Aug–Sep 2026 | — | — | — | `[LOG]` |

---

## 4. Major Audit Flags Discovered During Phase 0

1. **Evaluation Metric Contradiction (`[LOG]`):** The paper claims HICO-DET Default setting (both human and object boxes $\text{IoU} \ge 0.5$). All three evaluation scripts (`vynix_fewshot_adapter.py:834-844`, `test_vynix_pretrained.py:165-170`, `run_full_test_ablation.py:57-64`) actually compute **image-level multi-label classification AP** using `positive_objects` (no bounding-box IoU matching against GT boxes).
2. **Rare/Non-Rare Split Contradiction (`[LOG]`):** All evaluation scripts define `rare_ids = {h for h in range(meta.num_classes) if h % 4 == 0}` (150 classes = 25% of 600), not the official 155 Rare ($<10$ training instances) / 445 Non-Rare HICO-DET split.
3. **Soft Gate Formula Contradiction (`[LOG]`):** `compute_soft_geometric_gate` (`vynix_fewshot_adapter.py:187`) computes $\exp(-(d_{\text{box}}/\sigma)^2)$, omitting the factor of $2$ in $2\sigma^2$ claimed in the paper.
4. **Adapter / MLP / Blending Contradictions (`[LOG]`):** Cache keys are 1536-D visual-only (`f_fused`), not 1792-D `[f_v; f_geom]`; `SpatialMLP` is `8 -> 64 (LayerNorm+ReLU) -> 600 (Sigmoid)` added as `0.5 * log(p + 1e-6)`, not `8 -> 128 -> 256` GELU; `alpha` and `beta` are learned `nn.Parameter`s initialized at `1.0` and `5.5` (not fixed `alpha=0.35`), combined additively (`clip_logits + alpha * cache_logits`) rather than convex `(1-alpha) S_vlm + alpha S_cache`.
