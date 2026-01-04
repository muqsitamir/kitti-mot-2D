from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

@dataclass
class KittiLabel:
    frame: int
    track_id: int
    cls: str
    bbox_xyxy: Tuple[float, float, float, float]  # l,t,r,b
    trunc: float = 0.0
    occ: int = 0
    alpha: float = 0.0
    score: Optional[float] = None

def read_kitti_tracking_labels(label_path: Path) -> Dict[int, List[KittiLabel]]:
    by_frame = defaultdict(list)
    if not label_path.exists():
        return {}

    with label_path.open("r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 10:
                continue
            frame = int(p[0])
            tid = int(p[1])
            cls = p[2]
            trunc = float(p[3])
            occ = int(float(p[4]))
            alpha = float(p[5])
            l, t, r, b = map(float, p[6:10])

            score = None
            if len(p) >= 18:
                score = float(p[17])

            by_frame[frame].append(KittiLabel(
                frame=frame, track_id=tid, cls=cls,
                bbox_xyxy=(l, t, r, b),
                trunc=trunc, occ=occ, alpha=alpha, score=score
            ))
    return dict(by_frame)

class KittiTrackingSequence:
    def __init__(self, kitti_root: str | Path, split: str, seq: str):
        """
        Expected structure:
          <root>/<split>/image_02/<seq>/*.png
          <root>/<split>/label_02/<seq>.txt   (only for training)
        """
        self.root = Path(kitti_root)
        self.split = split
        self.seq = seq

        self.img_dir = self.root / split / "image_02" / seq
        self.label_path = self.root / split / "label_02" / f"{seq}.txt"

        self.frames = sorted(self.img_dir.glob("*.png"))
        self.labels_by_frame = read_kitti_tracking_labels(self.label_path) if self.label_path.exists() else {}

    def __len__(self) -> int:
        return len(self.frames)

    def frame_id(self, idx: int) -> int:
        return int(self.frames[idx].stem)

    def get_labels(self, frame_id: int, keep_classes=None) -> List[KittiLabel]:
        labs = self.labels_by_frame.get(frame_id, [])
        if keep_classes is not None:
            labs = [x for x in labs if x.cls in keep_classes]
        return labs