from __future__ import annotations
import argparse, json
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data_tracking_image_2")
    ap.add_argument("--split", choices=["training", "testing"], default="training")
    ap.add_argument("--seq", type=str, required=True)  # e.g. 0000
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--model", type=str, default="yolov8n.pt")  # fast
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / args.split / "image_02" / args.seq
    assert img_dir.exists(), f"Missing: {img_dir}"

    out_path = Path(args.out) if args.out else Path("assets") / f"yolo_{args.split}_{args.seq}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names  # dict[int,str]

    # COCO -> KITTI-ish mapping (keep it simple for your demo)
    # KITTI eval class names: Car, Pedestrian.
    map_to = {
        "car": "Car",
        "person": "Pedestrian",
        # # optional “vehicle-ish” mapped to Car:
        # "truck": "Car",
        # "bus": "Car",
    }

    frames = sorted(img_dir.glob("*.png"))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    with out_path.open("w") as f:
        for frame_idx, fp in tqdm(list(enumerate(frames)), desc=f"YOLO {args.split} {args.seq}"):
            img = cv2.imread(str(fp))
            if img is None:
                continue

            r = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
                name = names.get(int(c), str(c))
                if name not in map_to:
                    continue

                det = {
                    "frame": int(frame_idx),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(sc),
                    "cls": map_to[name],
                }
                f.write(json.dumps(det) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()