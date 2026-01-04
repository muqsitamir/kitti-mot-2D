from __future__ import annotations
import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kitti_loader import KittiTrackingSequence
from tracker import SortTracker
from metrics import eval_mot
from run_tracker_seq import make_detections_from_gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--jitter", type=float, default=2.0)
    ap.add_argument("--iou_match_thresh", type=float, default=0.5)
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.3)
    ap.add_argument("--max_age", type=int, default=10)
    ap.add_argument("--min_hits", type=int, default=3)
    args = ap.parse_args()

    ds = KittiTrackingSequence(args.root, "training", args.seq)
    keep = {"Car", "Pedestrian", "Van", "Cyclist"}

    tracker = SortTracker(
        iou_threshold=args.iou_assoc_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits
    )

    frames_gt = []
    frames_pr = []

    n = min(len(ds), args.max_frames)
    for i in range(n):
        fid = ds.frame_id(i)
        gt_labels = ds.get_labels(fid, keep_classes=keep)

        # GT format for motmetrics
        gt_frame = [{"track_id": l.track_id, "bbox_xyxy": l.bbox_xyxy, "cls": l.cls} for l in gt_labels]

        # detections (no GT IDs)
        dets = make_detections_from_gt(gt_labels, drop=args.drop, jitter=args.jitter, seed=123 + i)

        pr_tracks = tracker.step(dets)
        pr_frame = [{"track_id": t["track_id"], "bbox_xyxy": t["bbox_xyxy"], "cls": t["cls"]} for t in pr_tracks]

        frames_gt.append(gt_frame)
        frames_pr.append(pr_frame)

    summary = eval_mot(frames_gt, frames_pr, iou_match_thresh=args.iou_match_thresh)
    print(f"\nSequence {args.seq} | drop={args.drop} jitter={args.jitter} "
          f"| assoc_iou>={args.iou_assoc_thresh} match_iou>={args.iou_match_thresh}\n")
    print(summary)

if __name__ == "__main__":
    main()