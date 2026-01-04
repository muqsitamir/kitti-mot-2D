from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_loader import KittiTrackingSequence
from tracker import SortTracker
from metrics import eval_mot


def load_jsonl_by_frame(path: Path):
    """Load cached YOLO detections. Keep score so we can tune thresholds later."""
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


def _bbox_h(bb):
    return float(bb[3]) - float(bb[1])

def eval_one_seq(
    root: str,
    seq: str,
    det_path: Path,
    conf: float,
    car_conf: float | None,
    ped_conf: float | None,
    min_h_car: int,
    min_h_ped: int,
    iou_assoc: float,
    max_age: int,
    min_hits: int,
    max_frames: int | None,
    iou_match_thresh: float,
):
    ds = KittiTrackingSequence(root, "training", seq)
    keep = {"Car", "Pedestrian"}  # GT classes

    dets_by_frame = load_jsonl_by_frame(det_path)

    tracker = SortTracker(
        iou_threshold=iou_assoc,
        max_age=max_age,
        min_hits=min_hits,
    )

    frames_gt, frames_pr = [], []
    n = min(len(ds), max_frames) if max_frames is not None else len(ds)

    for i in range(n):
        fid = ds.frame_id(i)
        gt_labels = ds.get_labels(fid, keep_classes=keep)
        gt_frame = [{"track_id": l.track_id, "bbox_xyxy": l.bbox_xyxy, "cls": l.cls} for l in gt_labels]

        # Filter detections per class so we can tune thresholds meaningfully
        feed = []
        for d in dets_by_frame.get(i, []):
            cls = d["cls"]
            if cls not in keep:
                continue

            sc = float(d.get("score", 1.0))
            thr = conf
            if cls == "Car" and car_conf is not None:
                thr = car_conf
            elif cls == "Pedestrian" and ped_conf is not None:
                thr = ped_conf

            if sc < thr:
                continue

            bb = d["bbox_xyxy"]
            if cls == "Car" and min_h_car > 0 and _bbox_h(bb) < min_h_car:
                continue
            if cls == "Pedestrian" and min_h_ped > 0 and _bbox_h(bb) < min_h_ped:
                continue

            feed.append({"bbox_xyxy": bb, "cls": cls})

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
    ap.add_argument("--car_conf", type=float, default=None, help="override conf for Car only")
    ap.add_argument("--ped_conf", type=float, default=None, help="override conf for Pedestrian only")
    ap.add_argument("--min_h_car", type=int, default=0, help="min bbox height for Car (pixels)")
    ap.add_argument("--min_h_ped", type=int, default=0, help="min bbox height for Pedestrian (pixels)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--iou_match_thresh", type=float, default=0.5)

    # tracker params (leave fixed for this “baseline across all sequences” run)
    ap.add_argument("--iou_assoc_thresh", type=float, default=0.2)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--min_hits", type=int, default=3)

    ap.add_argument("--save_csv", type=str, default="assets/eval_yolo_all.csv")
    ap.add_argument("--seqs", type=str, default=None,
                    help="Comma-separated list like 0000,0003,0007 or leave empty for all 0000-0020")
    args = ap.parse_args()

    assets = Path(args.assets)

    if args.seqs:
        seq_list = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        seq_list = [f"{i:04d}" for i in range(21)]

    rows = []
    for seq in seq_list:
        det_path = assets / f"yolo_training_{seq}.jsonl"
        if not det_path.exists():
            print(f"[SKIP] missing dets: {det_path}")
            continue

        row = eval_one_seq(
            root=args.root,
            seq=seq,
            det_path=det_path,
            conf=args.conf,
            car_conf=args.car_conf,
            ped_conf=args.ped_conf,
            min_h_car=args.min_h_car,
            min_h_ped=args.min_h_ped,
            iou_assoc=args.iou_assoc_thresh,
            max_age=args.max_age,
            min_hits=args.min_hits,
            max_frames=args.max_frames,
            iou_match_thresh=args.iou_match_thresh,
        )
        row["seq"] = seq
        rows.append(row)
        print(
            f"[OK] seq {seq}  mota={row['mota']:.3f}  idf1={row['idf1']:.3f}  ids={int(row['num_switches'])} "
            f"| conf={args.conf} car_conf={args.car_conf} ped_conf={args.ped_conf} "
            f"| min_h_car={args.min_h_car} min_h_ped={args.min_h_ped} "
            f"| assoc_iou={args.iou_assoc_thresh} max_age={args.max_age} min_hits={args.min_hits}"
        )

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