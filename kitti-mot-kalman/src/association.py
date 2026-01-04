from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)

def iou_matrix(trk_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """
    trk_boxes: (T,4), det_boxes: (D,4) in xyxy
    returns IoU (T,D)
    """
    T = trk_boxes.shape[0]
    D = det_boxes.shape[0]
    out = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        for d in range(D):
            out[t, d] = iou_xyxy(trk_boxes[t], det_boxes[d])
    return out

def hungarian_iou_match(
    trk_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Returns:
      matches: list of (trk_idx, det_idx)
      unmatched_trks: list of trk indices
      unmatched_dets: list of det indices
    """
    T = trk_boxes.shape[0]
    D = det_boxes.shape[0]
    if T == 0:
        return [], [], list(range(D))
    if D == 0:
        return [], list(range(T)), []

    ious = iou_matrix(trk_boxes, det_boxes)
    cost = 1.0 - ious  # minimize cost

    trk_idx, det_idx = linear_sum_assignment(cost)

    matches = []
    unmatched_trks = set(range(T))
    unmatched_dets = set(range(D))

    for t, d in zip(trk_idx.tolist(), det_idx.tolist()):
        if ious[t, d] >= iou_threshold:
            matches.append((t, d))
            unmatched_trks.discard(t)
            unmatched_dets.discard(d)

    return matches, sorted(unmatched_trks), sorted(unmatched_dets)