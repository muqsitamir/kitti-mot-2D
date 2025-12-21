from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1] / "src"  # kitti-mot-kalman/src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from kitti_loader import KittiTrackingSequence
from viz import write_gt_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--out", type=str, default="assets/gt_0000.mp4")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cars_peds_only", action="store_true")
    args = ap.parse_args()

    ds = KittiTrackingSequence(args.root, args.split, args.seq)
    keep = {"Car", "Pedestrian"} if args.cars_peds_only else None

    out_path = write_gt_video(ds, args.out, max_frames=args.max_frames, fps=args.fps, keep_classes=keep)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()