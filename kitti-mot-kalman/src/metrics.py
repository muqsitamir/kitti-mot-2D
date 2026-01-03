from __future__ import annotations
from typing import List, Tuple
import numpy as np
import motmetrics as mm

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)

def iou_cost_matrix(gt_boxes: List[Tuple[float,float,float,float]],
                    pr_boxes: List[Tuple[float,float,float,float]],
                    iou_match_thresh: float = 0.5) -> np.ndarray:
    """
    motmetrics expects a distance matrix where:
      - lower is better
      - NaN means 'cannot match'
    We'll use distance = 1 - IoU, and disallow matches with IoU < threshold.
    """
    G, P = len(gt_boxes), len(pr_boxes)
    if G == 0 or P == 0:
        return np.empty((G, P), dtype=float)

    D = np.zeros((G, P), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pr_boxes):
            iou = iou_xyxy(g, p)
            if iou < iou_match_thresh:
                D[i, j] = np.nan
            else:
                D[i, j] = 1.0 - iou
    return D

def eval_mot(frames_gt, frames_pr, iou_match_thresh: float = 0.5):
    """
    frames_gt: list of frames, each is list of dicts {track_id:int, bbox_xyxy:(l,t,r,b)}
    frames_pr: same structure for predictions
    """
    acc = mm.MOTAccumulator(auto_id=True)

    for gt, pr in zip(frames_gt, frames_pr):
        gt_ids = [int(x["track_id"]) for x in gt]
        gt_boxes = [tuple(x["bbox_xyxy"]) for x in gt]

        pr_ids = [int(x["track_id"]) for x in pr]
        pr_boxes = [tuple(x["bbox_xyxy"]) for x in pr]

        dist = iou_cost_matrix(gt_boxes, pr_boxes, iou_match_thresh=iou_match_thresh)
        acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_matches"],
        name="seq",
    )
    return summary
