# kitti-mot-kalman/scripts/run_tracker_seq.py
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from kitti_loader import KittiTrackingSequence
from viz import draw_labels
from tracker import SortTracker

def make_detections(gt_labels, drop=0.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dets = []
    for lab in gt_labels:
        l, t, r, b = lab.bbox_xyxy
        if drop > 0 and rng.random() < drop:
            continue
        if jitter > 0:
            l += rng.normal(0, jitter); t += rng.normal(0, jitter)
            r += rng.normal(0, jitter); b += rng.normal(0, jitter)
        dets.append({"bbox_xyxy": (l, t, r, b), "cls": lab.cls})
    return dets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--out", type=str, default="assets/pred_0000.mp4")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=2.0)
    args = ap.parse_args()

    ds = KittiTrackingSequence(args.root, "training", args.seq)
    keep = {"Car", "Pedestrian"}

    # video writer
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(ds.frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    tracker = SortTracker(iou_threshold=0.3, max_age=10, min_hits=3)

    n = min(len(ds), args.max_frames)
    for i in range(n):
        frame_id = ds.frame_id(i)
        img = cv2.imread(str(ds.frames[i]))

        gt = ds.get_labels(frame_id, keep_classes=keep)
        dets = make_detections(gt, drop=args.drop, jitter=args.jitter, seed=42 + i)

        pred_tracks = tracker.step(dets)

        # draw predicted tracks using the same draw_labels util (dict-compatible)
        vis = draw_labels(img, pred_tracks, show_cls=True)
        writer.write(vis)

    writer.release()
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()