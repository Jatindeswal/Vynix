"""
Standard Box-Level HICO-DET Evaluator (Default Setting)
======================================================
Protocol:
- A prediction is a tuple: (human_box, object_box, hoi_id, score) in an image.
- A prediction is a true positive (TP) iff:
    min(IoU(human_box, gt_human_box), IoU(object_box, gt_object_box)) >= 0.5
  with an unmatched ground-truth pair of the same HOI class in the same image.
- Duplicate detections matching an already-matched GT are marked as false positives (FP).
- AP is computed per HOI class (600 classes total) using continuous interpolated precision-recall area.
- Full mAP: mean over all 600 HOI classes.
- Rare mAP: mean over the standard 138 rare classes (<10 training instances).
- Non-Rare mAP: mean over the standard 462 non-rare classes (>=10 training instances).
- Also provides image-level AP evaluation side by side for direct comparison against legacy metrics.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / max(union, 1e-8)


def compute_continuous_ap(tp_flags: np.ndarray, n_pos: int) -> float:
    """
    Computes standard continuous interpolated Average Precision (P-R AUC).
    Matches official PASCAL VOC / COCO / HICO-DET continuous metric.
    """
    if n_pos == 0:
        return 0.0
    if len(tp_flags) == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)
    prec = tp_cumsum / (tp_cumsum + fp_cumsum)
    rec = tp_cumsum / n_pos

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class HICODetBoxEvaluator:
    """
    Official Box-Level Default Setting Evaluator for HICO-DET.
    """

    def __init__(
        self,
        num_classes: int = 600,
        iou_thresh: float = 0.5,
        split_json_path: Optional[str] = None,
    ):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

        # Load standard splits
        if split_json_path and os.path.exists(split_json_path):
            with open(split_json_path, "r", encoding="utf-8") as f:
                splits = json.load(f)
            self.rare_ids = set(splits.get("rare", []))
            self.non_rare_ids = set(splits.get("non_rare", []))
        else:
            # Fallback will be populated externally or standard
            self.rare_ids = set()
            self.non_rare_ids = set()

        # ground_truths: class_id -> img_idx -> list of (h_box, o_box)
        self.ground_truths: Dict[int, Dict[int, List[Tuple[List[float], List[float]]]]] = {
            c: {} for c in range(num_classes)
        }
        # predictions: class_id -> list of (score, img_idx, h_box, o_box)
        self.predictions: Dict[int, List[Tuple[float, int, List[float], List[float]]]] = {
            c: [] for c in range(num_classes)
        }
        # Total positive GT annotations per class
        self.num_gt_per_class = np.zeros(num_classes, dtype=np.int32)

        # Image-level tracking for side-by-side comparison
        self.img_level_gt: Dict[int, Set[int]] = {}
        self.img_level_preds: Dict[int, Dict[int, float]] = {}

    def add_ground_truth(
        self,
        img_idx: int,
        gt_objects: List[Dict],
    ):
        """
        gt_objects: list of dicts from dataset parquet 'objects' column:
          {'id': <1-based class id>, 'bbox_human': [xmin, xmax, ymin, ymax], 'bbox_object': [xmin, xmax, ymin, ymax]}
        """
        pos_classes = set()
        for obj in gt_objects:
            c = obj["id"] - 1  # 1-indexed to 0-indexed
            if c < 0 or c >= self.num_classes:
                continue

            hb = obj.get("bbox_human")
            ob = obj.get("bbox_object")
            if hb is None or ob is None or len(hb) < 4 or len(ob) < 4:
                continue

            # Convert [xmin, xmax, ymin, ymax] to [xmin, ymin, xmax, ymax]
            h_box = [float(hb[0]), float(hb[2]), float(hb[1]), float(hb[3])]
            o_box = [float(ob[0]), float(ob[2]), float(ob[1]), float(ob[3])]

            if img_idx not in self.ground_truths[c]:
                self.ground_truths[c][img_idx] = []
            self.ground_truths[c][img_idx].append((h_box, o_box))
            self.num_gt_per_class[c] += 1
            pos_classes.add(c)

        self.img_level_gt[img_idx] = pos_classes

    def add_predictions(
        self,
        img_idx: int,
        triplets: List[Tuple[List[float], List[float], int, float]],
    ):
        """
        triplets: list of (h_box, o_box, hoi_id, score)
          where h_box and o_box are [xmin, ymin, xmax, ymax]
        """
        img_preds = {}
        for h_box, o_box, c, score in triplets:
            if c < 0 or c >= self.num_classes:
                continue
            self.predictions[c].append((float(score), img_idx, h_box, o_box))
            if c not in img_preds or score > img_preds[c]:
                img_preds[c] = float(score)

        self.img_level_preds[img_idx] = img_preds

    def evaluate_box_level(self, only_valid_gt: bool = False) -> Dict[str, Any]:
        """
        Evaluates box-level Default setting mAP:
        Prediction matches GT iff min(IoU_h, IoU_o) >= 0.5.
        If only_valid_gt=True, averages mAP strictly over classes with >=1 GT instance.
        Returns:
            Dict containing mAP_full, mAP_rare, mAP_non_rare, class counts, per_class_ap, num_gt_per_class.
        """
        per_class_ap = np.zeros(self.num_classes, dtype=np.float64)

        for c in range(self.num_classes):
            n_pos = self.num_gt_per_class[c]
            preds_c = self.predictions[c]

            if n_pos == 0:
                per_class_ap[c] = 0.0
                continue
            if len(preds_c) == 0:
                per_class_ap[c] = 0.0
                continue

            # Sort predictions descending by score
            preds_c.sort(key=lambda x: x[0], reverse=True)

            # Track matched GT instances: img_idx -> set of matched GT indices
            matched_gt: Dict[int, Set[int]] = {}
            tp_flags = np.zeros(len(preds_c), dtype=np.int32)

            for i, (score, img_idx, pred_h, pred_o) in enumerate(preds_c):
                gts_in_img = self.ground_truths[c].get(img_idx, [])
                if not gts_in_img:
                    tp_flags[i] = 0
                    continue

                if img_idx not in matched_gt:
                    matched_gt[img_idx] = set()

                best_match_idx = -1
                best_min_iou = -1.0

                for g_idx, (gt_h, gt_o) in enumerate(gts_in_img):
                    if g_idx in matched_gt[img_idx]:
                        continue
                    iou_h = compute_box_iou(pred_h, gt_h)
                    iou_o = compute_box_iou(pred_o, gt_o)
                    min_iou = min(iou_h, iou_o)

                    if min_iou >= self.iou_thresh and min_iou > best_min_iou:
                        best_min_iou = min_iou
                        best_match_idx = g_idx

                if best_match_idx >= 0:
                    matched_gt[img_idx].add(best_match_idx)
                    tp_flags[i] = 1
                else:
                    tp_flags[i] = 0

            per_class_ap[c] = compute_continuous_ap(tp_flags, n_pos)

        if only_valid_gt:
            valid_full = [c for c in range(self.num_classes) if self.num_gt_per_class[c] >= 1]
            valid_rare = [c for c in self.rare_ids if c < self.num_classes and self.num_gt_per_class[c] >= 1]
            valid_non_rare = [c for c in self.non_rare_ids if c < self.num_classes and self.num_gt_per_class[c] >= 1]

            map_full = float(np.mean([per_class_ap[c] for c in valid_full])) * 100.0 if valid_full else 0.0
            map_rare = float(np.mean([per_class_ap[c] for c in valid_rare])) * 100.0 if valid_rare else 0.0
            map_non_rare = float(np.mean([per_class_ap[c] for c in valid_non_rare])) * 100.0 if valid_non_rare else 0.0

            n_full = len(valid_full)
            n_rare = len(valid_rare)
            n_non_rare = len(valid_non_rare)
        else:
            rare_aps = [per_class_ap[c] for c in self.rare_ids if c < self.num_classes]
            non_rare_aps = [per_class_ap[c] for c in self.non_rare_ids if c < self.num_classes]

            map_full = float(np.mean(per_class_ap)) * 100.0
            map_rare = float(np.mean(rare_aps)) * 100.0 if rare_aps else 0.0
            map_non_rare = float(np.mean(non_rare_aps)) * 100.0 if non_rare_aps else 0.0

            n_full = self.num_classes
            n_rare = len(self.rare_ids)
            n_non_rare = len(self.non_rare_ids)

        return {
            "mAP_full": map_full,
            "mAP_rare": map_rare,
            "mAP_non_rare": map_non_rare,
            "num_classes_full": n_full,
            "num_classes_rare": n_rare,
            "num_classes_non_rare": n_non_rare,
            "per_class_ap": per_class_ap.tolist(),
            "num_gt_per_class": self.num_gt_per_class.tolist(),
        }

    def evaluate_image_level(self, total_images: int) -> Dict[str, float]:
        """
        Evaluates legacy image-level multi-label AP (ignoring bounding boxes).
        """
        per_class_ap = np.zeros(self.num_classes, dtype=np.float64)

        for c in range(self.num_classes):
            scores = np.array([self.img_level_preds.get(i, {}).get(c, 0.0) for i in range(total_images)])
            labels = np.array([1 if c in self.img_level_gt.get(i, set()) else 0 for i in range(total_images)])
            n_pos = int(labels.sum())
            if n_pos == 0:
                per_class_ap[c] = 0.0
                continue
            idx = np.argsort(-scores)
            tp_flags = labels[idx]
            per_class_ap[c] = compute_continuous_ap(tp_flags, n_pos)

        rare_aps = [per_class_ap[c] for c in self.rare_ids if c < self.num_classes]
        non_rare_aps = [per_class_ap[c] for c in self.non_rare_ids if c < self.num_classes]

        map_full = float(np.mean(per_class_ap)) * 100.0
        map_rare = float(np.mean(rare_aps)) * 100.0 if rare_aps else 0.0
        map_non_rare = float(np.mean(non_rare_aps)) * 100.0 if non_rare_aps else 0.0

        return {
            "mAP_full": map_full,
            "mAP_rare": map_rare,
            "mAP_non_rare": map_non_rare,
            "per_class_ap": per_class_ap.tolist(),
        }
