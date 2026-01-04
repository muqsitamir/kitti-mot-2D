from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import inspect

import cv2
import numpy as np

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
            by_frame[int(d["frame"])].append(
                {
                    "bbox_xyxy": tuple(d["bbox_xyxy"]),
                    "cls": d["cls"],
                    "score": float(d.get("score", 1.0)),
                }
            )
    return by_frame


def make_detections_from_gt(gt_labels, drop=0.0, jitter=0.0, seed=0):
    """Convert GT label objects -> det dicts (no track_id)."""
    rng = np.random.default_rng(seed)
    dets = []
    for lab in gt_labels:
        l, t, r, b = lab.bbox_xyxy
        if drop > 0 and rng.random() < drop:
            continue
        if jitter > 0:
            l += rng.normal(0, jitter)
            t += rng.normal(0, jitter)
            r += rng.normal(0, jitter)
            b += rng.normal(0, jitter)
        dets.append({"bbox_xyxy": (l, t, r, b), "cls": lab.cls})
    return dets


def bbox_h(bb):
    return bb[3] - bb[1]


def parse_keep_classes(s: str):
    # e.g. "Car,Pedestrian,Van,Cyclist"
    items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items)


def make_tracker(args):
    """Create SortTracker, optionally passing output_age if tracker supports it."""
    sig = inspect.signature(SortTracker.__init__)
    kwargs = dict(iou_threshold=args.iou_assoc_thresh, max_age=args.max_age, min_hits=args.min_hits)
    if "output_age" in sig.parameters:
        kwargs["output_age"] = args.output_age
    return SortTracker(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--seq", type=str, default="0000")
    ap.add_argument("--split", choices=["training", "testing"], default="training")
    ap.add_argument("--source", choices=["gt", "yolo"], default="gt")
    ap.add_argument("--out", type=str, default="assets/pred.mp4")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=10)

    # classes to load/track
    ap.add_argument("--keep_classes", type=str, default="Car,Pedestrian,Van,Cyclist")

    # GT-as-detections knobs
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)

    # YOLO cached dets knobs
    ap.add_argument("--dets", type=str, default=None, help="path to cached YOLO jsonl")
    ap.add_argument("--conf", type=float, default=0.25, help="global score filter (YOLO)")
    ap.add_argument("--car_conf", type=float, default=None, help="override conf for Car only")
    ap.add_argument("--ped_conf", type=float, default=None, help="override conf for Pedestrian only")
    ap.add_argument("--min_h_car", type=int, default=0)
    ap.add_argument("--min_h_ped", type=int, default=0)

    # tracker params
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.2)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--output_age", type=int, default=0, help="only used if your tracker supports it")

    # debug
    ap.add_argument("--debug_first_n", type=int, default=0)

    args = ap.parse_args()
    keep = parse_keep_classes(args.keep_classes)

    ds = KittiTrackingSequence(args.root, args.split, args.seq)

    # Load YOLO dets if needed
    dets_by_frame = None
    if args.source == "yolo":
        det_path = Path(args.dets) if args.dets else Path("assets") / f"yolo_{args.split}_{args.seq}.jsonl"
        assert det_path.exists(), f"Missing detections file: {det_path}. Run cache_yolo_dets.py first."
        dets_by_frame = load_jsonl_by_frame(det_path)

    # Video writer
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(ds.frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {ds.frames[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    tracker = make_tracker(args)

    n = min(len(ds), args.max_frames)
    for i in range(n):
        frame_id = ds.frame_id(i)
        img = cv2.imread(str(ds.frames[i]))
        if img is None:
            continue

        if args.source == "gt":
            gt = ds.get_labels(frame_id, keep_classes=keep)
            dets = make_detections_from_gt(gt, drop=args.drop, jitter=args.jitter, seed=42 + i)

            tracks = tracker.step(dets)

            if i < args.debug_first_n:
                gt_counts = {}
                for g in gt:
                    gt_counts[g.cls] = gt_counts.get(g.cls, 0) + 1
                det_counts = {}
                for d in dets:
                    det_counts[d["cls"]] = det_counts.get(d["cls"], 0) + 1
                trk_counts = {}
                for t in tracks:
                    trk_counts[t["cls"]] = trk_counts.get(t["cls"], 0) + 1
                print(f"[frame {i:04d}] GT={len(gt)} {gt_counts} | dets={len(dets)} {det_counts} | tracks={len(tracks)} {trk_counts}")

        else:  # yolo
            frame_dets_raw = dets_by_frame.get(i, [])
            dets = []
            for d in frame_dets_raw:
                cls = d["cls"]
                bb = d["bbox_xyxy"]
                sc = d["score"]

                # per-class thresholds if provided
                thr = args.conf
                if cls == "Car" and args.car_conf is not None:
                    thr = args.car_conf
                if cls == "Pedestrian" and args.ped_conf is not None:
                    thr = args.ped_conf

                if sc < thr:
                    continue

                # per-class min-height filters
                if cls == "Car" and args.min_h_car > 0 and bbox_h(bb) < args.min_h_car:
                    continue
                if cls == "Pedestrian" and args.min_h_ped > 0 and bbox_h(bb) < args.min_h_ped:
                    continue

                # keep only selected classes
                if cls not in keep:
                    continue

                dets.append({"bbox_xyxy": bb, "cls": cls})

            tracks = tracker.step(dets)

            if i < args.debug_first_n:
                det_counts = {}
                for d in dets:
                    det_counts[d["cls"]] = det_counts.get(d["cls"], 0) + 1
                trk_counts = {}
                for t in tracks:
                    trk_counts[t["cls"]] = trk_counts.get(t["cls"], 0) + 1
                print(f"[frame {i:04d}] dets={len(dets)} {det_counts} | tracks={len(tracks)} {trk_counts}")

        vis = draw_labels(img, tracks, show_cls=True)
        writer.write(vis)

    writer.release()
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()