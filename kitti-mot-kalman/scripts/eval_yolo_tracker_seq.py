from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_loader import KittiTrackingSequence
from tracker import SortTracker
from metrics import eval_mot


def load_yolo_by_frame(jsonl_path: Path):
    by_frame = defaultdict(list)
    with jsonl_path.open("r") as f:
        for line in f:
            d = json.loads(line)
            by_frame[int(d["frame"])].append(d)
    return by_frame


def h(bb):  # bbox height
    return bb[3] - bb[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--max_frames", type=int, default=300)

    ap.add_argument("--dets", type=str, default=None, help="assets/yolo_training_XXXX.jsonl")
    ap.add_argument("--iou_match_thresh", type=float, default=0.5)

    # class-specific detection filtering (recommended)
    ap.add_argument("--car_conf", type=float, default=0.6)
    ap.add_argument("--ped_conf", type=float, default=0.35)
    ap.add_argument("--min_h_car", type=int, default=25)
    ap.add_argument("--min_h_ped", type=int, default=25)

    # tracker params
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.2)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--output_age", type=int, default=2)

    args = ap.parse_args()

    ds = KittiTrackingSequence(args.root, "training", args.seq)
    keep = {"Car", "Pedestrian"}

    det_path = Path(args.dets) if args.dets else Path("assets") / f"yolo_training_{args.seq}.jsonl"
    assert det_path.exists(), f"Missing dets file: {det_path}. Run cache_yolo_dets.py first."
    dets_by_frame = load_yolo_by_frame(det_path)

    tracker = SortTracker(
        iou_threshold=args.iou_assoc_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        output_age=args.output_age,
    )

    frames_gt, frames_pr = [], []
    n = min(len(ds), args.max_frames)

    for i in range(n):
        fid = ds.frame_id(i)
        gt_labels = ds.get_labels(fid, keep_classes=keep)
        gt_frame = [{"track_id": l.track_id, "bbox_xyxy": l.bbox_xyxy, "cls": l.cls} for l in gt_labels]

        # build tracker input from YOLO dets (with per-class conf + size filters)
        feed = []
        for d in dets_by_frame.get(i, []):
            cls = d["cls"]
            bb = tuple(d["bbox_xyxy"])
            sc = float(d.get("score", 1.0))

            if cls == "Car":
                if sc < args.car_conf or h(bb) < args.min_h_car:
                    continue
            elif cls == "Pedestrian":
                if sc < args.ped_conf or h(bb) < args.min_h_ped:
                    continue
            else:
                continue

            feed.append({"bbox_xyxy": bb, "cls": cls})

        pr_tracks = tracker.step(feed)
        pr_frame = [{"track_id": t["track_id"], "bbox_xyxy": t["bbox_xyxy"], "cls": t["cls"]} for t in pr_tracks]

        frames_gt.append(gt_frame)
        frames_pr.append(pr_frame)

    summary = eval_mot(frames_gt, frames_pr, iou_match_thresh=args.iou_match_thresh)

    print(
        f"\nYOLO→Tracker | seq={args.seq} frames={n} "
        f"| car_conf>={args.car_conf} ped_conf>={args.ped_conf} "
        f"| assoc_iou>={args.iou_assoc_thresh} match_iou>={args.iou_match_thresh}\n"
    )
    print(summary)


if __name__ == "__main__":
    main()