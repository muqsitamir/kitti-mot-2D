from __future__ import annotations
from pathlib import Path
import cv2


def draw_labels(img, labels, show_cls=True):
    out = img.copy()

    for lab in labels:
        # Support both dataclass-like labels and dict outputs from tracker
        if isinstance(lab, dict):
            l, t, r, b = lab["bbox_xyxy"]
            cls = lab.get("cls", "obj")
            tid = lab.get("track_id", "-")
        else:
            l, t, r, b = lab.bbox_xyxy
            cls = getattr(lab, "cls", "obj")
            tid = getattr(lab, "track_id", "-")

        l, t, r, b = int(l), int(t), int(r), int(b)
        cv2.rectangle(out, (l, t), (r, b), (0, 255, 0), 2)

        txt = f"{tid}" if not show_cls else f"{cls}:{tid}"
        cv2.putText(out, txt, (l, max(0, t - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return out

def write_gt_video(seq_ds, out_path: str | Path, max_frames=None, fps=10, keep_classes=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # read first frame to get size
    first = cv2.imread(str(seq_ds.frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = len(seq_ds) if max_frames is None else min(len(seq_ds), max_frames)
    for i in range(n):
        frame_id = seq_ds.frame_id(i)
        img = cv2.imread(str(seq_ds.frames[i]))
        labs = seq_ds.get_labels(frame_id, keep_classes=keep_classes)
        vis = draw_labels(img, labs, show_cls=True)
        writer.write(vis)

    writer.release()
    return out_path