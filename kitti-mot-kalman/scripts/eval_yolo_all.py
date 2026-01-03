from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_loader import KittiTrackingSequence
from tracker import SortTracker
from metrics import eval_mot


def load_jsonl_by_frame(path: Path, min_conf: float):
    by_frame = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            d = json.loads(line)
            sc = float(d.get("score", 1.0))
            if sc < min_conf:
                continue
            by_frame[int(d["frame"])].append({
                "bbox_xyxy": tuple(d["bbox_xyxy"]),
                "cls": d["cls"],
            })
    return by_frame


def eval_one_seq(root: str, seq: str, det_path: Path, conf: float,
                 iou_assoc: float, max_age: int, min_hits: int, output_age: int,
                 max_frames: int | None, iou_match_thresh: float):
    ds = KittiTrackingSequence(root, "training", seq)
    keep = {"Car", "Pedestrian"}  # GT classes

    dets_by_frame = load_jsonl_by_frame(det_path, conf)

    tracker = SortTracker(
        iou_threshold=iou_assoc,
        max_age=max_age,
        min_hits=min_hits,
        output_age=output_age,
    )

    frames_gt, frames_pr = [], []
    n = min(len(ds), max_frames) if max_frames is not None else len(ds)

    for i in range(n):
        fid = ds.frame_id(i)
        gt_labels = ds.get_labels(fid, keep_classes=keep)
        gt_frame = [{"track_id": l.track_id, "bbox_xyxy": l.bbox_xyxy, "cls": l.cls} for l in gt_labels]

        feed = dets_by_frame.get(i, [])
        pr_tracks = tracker.step(feed)
        pr_frame = [{"track_id": t["track_id"], "bbox_xyxy": t["bbox_xyxy"], "cls": t["cls"]} for t in pr_tracks]

        frames_gt.append(gt_frame)
        frames_pr.append(pr_frame)

    summary = eval_mot(frames_gt, frames_pr, iou_match_thresh=iou_match_thresh)
    row = summary.iloc[0].to_dict()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--assets", type=str, default="assets")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--iou_match_thresh", type=float, default=0.5)

    # tracker params (leave fixed for this “baseline across all sequences” run)
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.2)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--output_age", type=int, default=2)

    ap.add_argument("--save_csv", type=str, default="assets/eval_yolo_all.csv")
    args = ap.parse_args()

    assets = Path(args.assets)

    rows = []
    for i in range(21):
        seq = f"{i:04d}"
        det_path = assets / f"yolo_training_{seq}.jsonl"
        if not det_path.exists():
            print(f"[SKIP] missing dets: {det_path}")
            continue

        row = eval_one_seq(
            root=args.root,
            seq=seq,
            det_path=det_path,
            conf=args.conf,
            iou_assoc=args.iou_assoc_thresh,
            max_age=args.max_age,
            min_hits=args.min_hits,
            output_age=args.output_age,
            max_frames=args.max_frames,
            iou_match_thresh=args.iou_match_thresh,
        )
        row["seq"] = seq
        rows.append(row)
        print(f"[OK] seq {seq}  mota={row['mota']:.3f}  idf1={row['idf1']:.3f}  ids={int(row['num_switches'])}")

    df = pd.DataFrame(rows).set_index("seq").sort_index()
    print("\nPer-sequence:")
    print(df)

    print("\nMean:")
    print(df.mean(numeric_only=True))

    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv)
    print(f"\nSaved: {args.save_csv}")


if __name__ == "__main__":
    main()