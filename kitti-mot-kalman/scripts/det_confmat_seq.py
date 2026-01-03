from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Debug-friendly imports
THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_loader import KittiTrackingSequence


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


def match_hungarian(gt_boxes, det_boxes, iou_thresh: float):
    """
    Returns list of (gt_idx, det_idx) matches with IoU >= iou_thresh
    """
    G, D = len(gt_boxes), len(det_boxes)
    if G == 0 or D == 0:
        return []

    ious = np.zeros((G, D), dtype=np.float32)
    for i in range(G):
        for j in range(D):
            ious[i, j] = iou_xyxy(gt_boxes[i], det_boxes[j])

    # maximize IoU -> minimize cost = 1 - IoU
    cost = 1.0 - ious
    gi, dj = linear_sum_assignment(cost)

    matches = []
    for g, d in zip(gi.tolist(), dj.tolist()):
        if ious[g, d] >= iou_thresh:
            matches.append((g, d))
    return matches


def load_yolo_jsonl_by_frame(path: Path, conf_thresh: float, min_h: int, keep_classes: set[str]):
    by_frame = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            d = json.loads(line)
            sc = float(d.get("score", 1.0))
            if sc < conf_thresh:
                continue
            cls = d["cls"]
            if cls not in keep_classes:
                continue
            x1, y1, x2, y2 = d["bbox_xyxy"]
            if (y2 - y1) < min_h:
                continue
            by_frame[int(d["frame"])].append(
                {"bbox_xyxy": (float(x1), float(y1), float(x2), float(y2)), "cls": cls, "score": sc}
            )
    return by_frame


def safe_prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--dets", type=str, default=None, help="assets/yolo_training_0000.jsonl")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min_h_car", type=int, default=25)
    ap.add_argument("--min_h_ped", type=int, default=40)
    args = ap.parse_args()

    keep = {"Car", "Pedestrian"}
    ds = KittiTrackingSequence(args.root, "training", args.seq)

    det_path = Path(args.dets) if args.dets else Path("assets") / f"yolo_training_{args.seq}.jsonl"
    assert det_path.exists(), f"Missing dets file: {det_path}"

    # Load dets twice with different size filters per class (simple + effective)
    dets_by_frame_car = load_yolo_jsonl_by_frame(det_path, args.conf, args.min_h_car, {"Car"})
    dets_by_frame_ped = load_yolo_jsonl_by_frame(det_path, args.conf, args.min_h_ped, {"Pedestrian"})

    # Stats
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in keep}
    # Confusion for matched pairs (GT vs Pred) — requires joint matching (regardless class)
    # We’ll do it by matching all boxes first, then counting label pairs for matches.
    conf_mat = {gt: {pr: 0 for pr in keep} for gt in keep}

    n = min(len(ds), args.max_frames)

    for i in range(n):
        fid = ds.frame_id(i)
        gt_labels = ds.get_labels(fid, keep_classes=keep)

        # Split GT by class
        gt_by_cls = {"Car": [], "Pedestrian": []}
        for g in gt_labels:
            gt_by_cls[g.cls].append(g.bbox_xyxy)

        # Split dets by class (with different min height)
        det_by_cls = {
            "Car": [d["bbox_xyxy"] for d in dets_by_frame_car.get(i, [])],
            "Pedestrian": [d["bbox_xyxy"] for d in dets_by_frame_ped.get(i, [])],
        }

        # Per-class TP/FP/FN
        for cls in keep:
            gt_boxes = gt_by_cls[cls]
            det_boxes = det_by_cls[cls]
            matches = match_hungarian(gt_boxes, det_boxes, args.iou)

            tp = len(matches)
            fp = len(det_boxes) - tp
            fn = len(gt_boxes) - tp

            stats[cls]["tp"] += tp
            stats[cls]["fp"] += fp
            stats[cls]["fn"] += fn

        # Optional: confusion matrix over matched pairs (ignoring class during matching)
        all_gt = [(g.cls, g.bbox_xyxy) for g in gt_labels]
        all_det = []
        for d in dets_by_frame_car.get(i, []):
            all_det.append(("Car", d["bbox_xyxy"]))
        for d in dets_by_frame_ped.get(i, []):
            all_det.append(("Pedestrian", d["bbox_xyxy"]))

        gt_boxes = [b for (_, b) in all_gt]
        det_boxes = [b for (_, b) in all_det]
        matches_all = match_hungarian(gt_boxes, det_boxes, args.iou)

        for gi, dj in matches_all:
            gt_cls = all_gt[gi][0]
            pr_cls = all_det[dj][0]
            if gt_cls in keep and pr_cls in keep:
                conf_mat[gt_cls][pr_cls] += 1

    # Print results
    print(f"\nDetection quality (YOLO) on seq={args.seq}, frames={n}, conf>={args.conf}, IoU>={args.iou}")
    print(f"Size filters: Car h>={args.min_h_car}px, Ped h>={args.min_h_ped}px\n")

    total_tp = total_fp = total_fn = 0
    for cls in ["Car", "Pedestrian"]:
        tp, fp, fn = stats[cls]["tp"], stats[cls]["fp"], stats[cls]["fn"]
        prec, rec, f1 = safe_prf(tp, fp, fn)
        total_tp += tp; total_fp += fp; total_fn += fn
        print(f"{cls:10s} TP={tp:5d}  FP={fp:5d}  FN={fn:5d}  |  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    prec, rec, f1 = safe_prf(total_tp, total_fp, total_fn)
    print(f"\nOVERALL     TP={total_tp:5d}  FP={total_fp:5d}  FN={total_fn:5d}  |  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    print("\nMatched-pair confusion (GT rows vs Pred cols) [only for matched IoU pairs]")
    print("            Car   Pedestrian")
    for gt in ["Car", "Pedestrian"]:
        print(f"{gt:10s} {conf_mat[gt]['Car']:5d} {conf_mat[gt]['Pedestrian']:11d}")


if __name__ == "__main__":
    main()