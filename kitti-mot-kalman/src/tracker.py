from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from association import hungarian_iou_match

def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h

def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (x1, y1, x2, y2)

class KalmanCV:
    def __init__(self, dt: float = 1.0):
        # state: [cx, cy, vx, vy, w, h]
        self.dt = dt
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 10.0

        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # measurement z: [cx, cy, w, h]
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 4] = 1.0
        self.H[3, 5] = 1.0

        # process + measurement noise (tunable)
        self.Q = np.diag([1.0, 1.0, 5.0, 5.0, 1.0, 1.0]).astype(np.float32)
        self.R = np.diag([10.0, 10.0, 25.0, 25.0]).astype(np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray):
        # z shape (4,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

@dataclass
class Track:
    track_id: int
    cls: str
    kf: KalmanCV
    hits: int = 0
    age: int = 0
    time_since_update: int = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det_bbox_xyxy):
        cx, cy, w, h = xyxy_to_cxcywh(det_bbox_xyxy)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0

    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        cx, cy, vx, vy, w, h = self.kf.x.flatten().tolist()
        return cxcywh_to_xyxy(cx, cy, w, h)

class SortTracker:
    def __init__(self, iou_threshold=0.3, max_age=10, min_hits=3, output_age=0):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1
        self.output_age = output_age

    def step(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        detections: [{ "bbox_xyxy": (l,t,r,b), "cls": "Car" }, ...]
        returns active tracks as list of dicts with bbox + id + cls
        """
        # 1) Predict existing tracks
        for trk in self.tracks:
            trk.predict()

        # 2) Build match problem (class-aware: match cars with cars, peds with peds)
        out_tracks: List[Track] = []
        used_det = set()

        # process per class separately for simplicity
        classes = sorted(set([d["cls"] for d in detections] + [t.cls for t in self.tracks]))
        new_tracks: List[Track] = []

        for cls in classes:
            trk_idxs = [i for i, t in enumerate(self.tracks) if t.cls == cls]
            det_idxs = [i for i, d in enumerate(detections) if d["cls"] == cls]

            trk_boxes = np.array([self.tracks[i].bbox_xyxy() for i in trk_idxs], dtype=np.float32) if trk_idxs else np.zeros((0,4), dtype=np.float32)
            det_boxes = np.array([detections[i]["bbox_xyxy"] for i in det_idxs], dtype=np.float32) if det_idxs else np.zeros((0,4), dtype=np.float32)

            matches, un_trk_local, un_det_local = hungarian_iou_match(trk_boxes, det_boxes, self.iou_threshold)

            # update matched
            for t_local, d_local in matches:
                t_idx = trk_idxs[t_local]
                d_idx = det_idxs[d_local]
                self.tracks[t_idx].update(detections[d_idx]["bbox_xyxy"])
                used_det.add(d_idx)

            # create new tracks for unmatched dets
            for d_local in un_det_local:
                d_idx = det_idxs[d_local]
                bb = detections[d_idx]["bbox_xyxy"]
                cx, cy, w, h = xyxy_to_cxcywh(bb)
                kf = KalmanCV(dt=1.0)
                kf.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]], dtype=np.float32)
                trk = Track(track_id=self._next_id, cls=cls, kf=kf, hits=1, age=1, time_since_update=0)
                self._next_id += 1
                new_tracks.append(trk)
                used_det.add(d_idx)

        self.tracks.extend(new_tracks)

        # 3) Kill old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 4) Output confirmed tracks (and optionally also recently updated ones)
        outputs = []
        for t in self.tracks:
            if t.hits >= self.min_hits and t.time_since_update == 0:
                outputs.append({"bbox_xyxy": t.bbox_xyxy(), "track_id": t.track_id, "cls": t.cls})
        return outputs

