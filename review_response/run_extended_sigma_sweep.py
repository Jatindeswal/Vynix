import os
import sys
import json
import time
from typing import List, Dict, Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

import torch
from vynix_fewshot_adapter import HOIMeta, compute_soft_geometric_gate
from review_response.hico_evaluator import HICODetBoxEvaluator


def evaluate_grid_point(
    cached_records: List[Dict],
    meta: HOIMeta,
    det_lambda: float,
    obj_gamma: float,
    sigma_contact: float,
    det_thresh: float = 0.08,
    split_json_path: str = "review_response/standard_hico_splits.json",
) -> Dict[str, float]:
    evaluator = HICODetBoxEvaluator(
        num_classes=meta.num_classes,
        iou_thresh=0.5,
        split_json_path=split_json_path,
    )

    for rec in cached_records:
        eval_idx = rec["eval_idx"]
        evaluator.add_ground_truth(eval_idx, rec["gt_objects"])

        candidate_pairs = [
            p for p in rec["pairs_data"]
            if p["p_conf"] >= det_thresh and p["o_conf"] >= det_thresh
        ]
        if len(candidate_pairs) > 50:
            candidate_pairs.sort(key=lambda x: x["p_conf"] * x["o_conf"], reverse=True)
            candidate_pairs = candidate_pairs[:50]

        triplets = []
        for p in candidate_pairs:
            p_conf, o_conf = p["p_conf"], p["o_conf"]
            p_box, o_box = p["p_box"], p["o_box"]
            iou_val, b_dist = p["iou_val"], p["b_dist"]
            det_score = (p_conf * o_conf) ** det_lambda
            obj_gate = (p["cos_sim"] ** obj_gamma) if obj_gamma > 0 else 1.0

            for verb, _, hoi_id in p["entries"]:
                raw_prob = p["base_probs"][hoi_id].item()
                gate = compute_soft_geometric_gate(verb, iou_val, b_dist, sigma_contact=sigma_contact)
                conf = det_score * raw_prob * gate * obj_gate
                triplets.append((p_box, o_box, hoi_id, conf))

        evaluator.add_predictions(eval_idx, triplets)

    return evaluator.evaluate_box_level(only_valid_gt=True)


def select_best_param(candidates: List[Tuple[float, Dict]], default_val: float) -> Tuple[float, float]:
    """
    Selection rule, fixed in advance:
    Pick the best-Full-mAP value; if within 0.10 of another, keep the one closest to current default.
    """
    max_map = max(m["mAP_full"] for _, m in candidates)
    close_candidates = [(v, m) for v, m in candidates if m["mAP_full"] >= max_map - 0.10]
    close_candidates.sort(key=lambda x: (round(abs(x[0] - default_val), 4), -x[1]["mAP_full"]))
    best_val, best_metrics = close_candidates[0]
    return best_val, best_metrics["mAP_full"]


def main():
    dataset_dir = "."
    meta = HOIMeta(dataset_dir)
    sigma_grid = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16]

    print("=" * 80)
    print("EXTENDED SIGMA GRID SWEEP ON VAL_10 (Objective: Full mAP, tie-break: <=0.10 from peak closest to 0.08)")
    print(f"Grid: {sigma_grid}")
    print("=" * 80)

    results_all = {}

    # --- Vynix 1.0 ---
    print("\nLoading val_cache_v1.0.pt...")
    v1_recs = torch.load("review_response/val_caches/val_cache_v1.0.pt", map_location="cpu")
    print(f"Loaded {len(v1_recs)} records.")

    print("\n--- Sweeping Sigma for Vynix 1.0 (lambda=1.0, gamma_obj=0.0, det_thresh=0.08) ---")
    candidates_v1 = []
    v1_sigma_results = []
    for s in sigma_grid:
        t0 = time.time()
        res = evaluate_grid_point(
            v1_recs, meta,
            det_lambda=1.0,
            obj_gamma=0.0,
            sigma_contact=s,
            det_thresh=0.08,
        )
        dt = time.time() - t0
        print(f"  sigma={s:4.2f} | Full ({res['num_classes_full']} cls): {res['mAP_full']:6.2f}% | "
              f"Rare ({res['num_classes_rare']} cls): {res['mAP_rare']:6.2f}% | "
              f"Non-Rare ({res['num_classes_non_rare']} cls): {res['mAP_non_rare']:6.2f}% | ({dt:.1f}s)")
        candidates_v1.append((s, res))
        v1_sigma_results.append({"sigma": s, "metrics": res})

    best_s_v1, best_map_v1 = select_best_param(candidates_v1, default_val=0.08)
    print(f"\n=> Vynix 1.0 Selected Sigma: {best_s_v1:.2f} (Full mAP: {best_map_v1:.2f}%)")
    results_all["vynix_1.0_extended_sigma"] = v1_sigma_results
    results_all["vynix_1.0_selected_sigma"] = best_s_v1

    # Clean memory
    del v1_recs

    # --- Vynix 2.0 ---
    print("\nLoading val_cache_v2.0.pt...")
    v2_recs = torch.load("review_response/val_caches/val_cache_v2.0.pt", map_location="cpu")
    print(f"Loaded {len(v2_recs)} records.")

    print("\n--- Sweeping Sigma for Vynix 2.0 (lambda=0.90, gamma_obj=0.10, det_thresh=0.08) ---")
    candidates_v2 = []
    v2_sigma_results = []
    for s in sigma_grid:
        t0 = time.time()
        res = evaluate_grid_point(
            v2_recs, meta,
            det_lambda=0.90,
            obj_gamma=0.10,
            sigma_contact=s,
            det_thresh=0.08,
        )
        dt = time.time() - t0
        print(f"  sigma={s:4.2f} | Full ({res['num_classes_full']} cls): {res['mAP_full']:6.2f}% | "
              f"Rare ({res['num_classes_rare']} cls): {res['mAP_rare']:6.2f}% | "
              f"Non-Rare ({res['num_classes_non_rare']} cls): {res['mAP_non_rare']:6.2f}% | ({dt:.1f}s)")
        candidates_v2.append((s, res))
        v2_sigma_results.append({"sigma": s, "metrics": res})

    best_s_v2, best_map_v2 = select_best_param(candidates_v2, default_val=0.08)
    print(f"\n=> Vynix 2.0 Selected Sigma: {best_s_v2:.2f} (Full mAP: {best_map_v2:.2f}%)")
    results_all["vynix_2.0_extended_sigma"] = v2_sigma_results
    results_all["vynix_2.0_selected_sigma"] = best_s_v2

    # Save to json
    with open("review_response/extended_sigma_results.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print("\nSaved extended sigma results to review_response/extended_sigma_results.json")


if __name__ == "__main__":
    main()
