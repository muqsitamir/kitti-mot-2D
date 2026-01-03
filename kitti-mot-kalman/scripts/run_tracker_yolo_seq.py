from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import cv2

THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_loader import KittiTrackingSequence
from tracker import SortTracker
from viz import draw_labels


def load_jsonl_by_frame(path: Path):
    by_frame = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            d = json.loads(line)
            by_frame[int(d["frame"])].append({
                "bbox_xyxy": tuple(d["bbox_xyxy"]),
                "cls": d["cls"],
                "score": float(d.get("score", 1.0)),
            })
    return by_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--split", choices=["training", "testing"], default="training")
    ap.add_argument("--dets", type=str, default=None, help="path to cached YOLO jsonl")
    ap.add_argument("--out", type=str, default="assets/pred_yolo_0000.mp4")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25, help="filter cached dets by score")
    # tracker params
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.1)
    ap.add_argument("--max_age", type=int, default=20)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--output_age", type=int, default=2)
    args = ap.parse_args()

    ds = KittiTrackingSequence(args.root, args.split, args.seq)

    det_path = Path(args.dets) if args.dets else Path("assets") / f"yolo_{args.split}_{args.seq}.jsonl"
    assert det_path.exists(), f"Missing detections file: {det_path}. Run cache_yolo_dets.py first."
    dets_by_frame = load_jsonl_by_frame(det_path)

    tracker = SortTracker(
        iou_threshold=args.iou_assoc_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        output_age=args.output_age,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(ds.frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    n = min(len(ds), args.max_frames)
    for i in range(n):
        img = cv2.imread(str(ds.frames[i]))
        frame_dets = [d for d in dets_by_frame.get(i, []) if d["score"] >= args.conf]

        # feed tracker ONLY bbox+cls
        feed = [{"bbox_xyxy": d["bbox_xyxy"], "cls": d["cls"]} for d in frame_dets]
        tracks = tracker.step(feed)

        vis = draw_labels(img, tracks, show_cls=True)  # draw_labels must support dicts
        writer.write(vis)

    writer.release()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()