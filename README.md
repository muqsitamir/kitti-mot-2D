
# KITTI Multi-Object Tracking (SORT-style Kalman + Hungarian)

A lightweight **tracking-by-detection** baseline for the **KITTI Tracking** dataset, implemented as a clean, learning-oriented **SORT-style** tracker.

**Core ideas**
- **Kalman Filter** per track (predict / update)
- **IoU-based data association** solved with the **Hungarian algorithm**
- Track lifecycle management (`min_hits`, `max_age`)
- Optional **YOLO detections** (cached to disk) for a realistic end-to-end run

## Goal

This implementation establishes a solid **baseline** for 2D multi-object tracking on KITTI:
- It proves the **end-to-end pipeline** (detections → association → tracks → metrics)
- It creates a foundation that you can extend with minimal code changes

Straightforward improvements (optional):
- Better detector / class mapping and thresholding
- Add **appearance features** (DeepSORT-style re-ID) for more robust association
- Motion model improvements (camera motion compensation, different state models)
- Smarter track management (confidence decay, class-consistent association)

---

## Demo (KITTI seq `0000`)

### YOLO → Tracker

![YOLO → SORT-style tracker demo (KITTI seq 0000)](assets/pred_yolo_0000.gif)


### GT-as-detections sanity check

![GT-as-detections → tracker demo (KITTI seq 0000)](assets/demo_gt_sort_0000.gif)

---

## Results

Mean metrics on **KITTI training split (21 sequences)** using **YOLO → Tracker** (cached detections + runtime filtering).

- Per-sequence breakdown CSV: `assets/eval_yolo_all.csv`

| Metric |       Mean |
|---|-----------:|
| **MOTA** | **0.4289** |
| **IDF1** | **0.5317** |
| **ID switches (avg)** |   **5.95** |
| **False positives (avg)** |  **65.48** |
| **Misses (avg)** | **864.62** |

---

## Project layout

```text
kitti-mot-kalman/
  scripts/        # runnable scripts (cache, run, evaluate)
  src/            # loader, tracker, association, metrics, visualization
assets/           # generated videos, cached detections, evaluation CSVs
data/             # KITTI tracking dataset (not included)
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want YOLO caching:

```bash
pip install ultralytics
```

---

## Data

Download the **KITTI Tracking** dataset (images + labels).

Expected structure:

```text
data/
  data_tracking_image_2/
    training/
      image_02/<SEQ>/*.png
      label_02/<SEQ>.txt
      calib/<SEQ>.txt
    testing/
      image_02/<SEQ>/*.png
      calib/<SEQ>.txt
```

Defaults assume:
- `--root data/data_tracking_image_2`

---

## Quickstart

> Most scripts expect imports from `kitti-mot-kalman/src`. The simplest way is setting `PYTHONPATH`.

### A) Sanity video: GT boxes as detections
Validates tracker logic without detector noise.

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/run_tracker_seq.py \
  --source gt --split training --seq 0010 \
  --max_frames 250 --fps 10 \
  --drop 0 --jitter 0 \
  --iou_assoc_thresh 0.10 --max_age 30 --min_hits 1 \
  --out assets/demo_gt_sort_0010.mp4
```

### B) Cache detections once, tune later

```bash
python kitti-mot-kalman/scripts/cache_yolo_dets.py \
  --split training --seq 0010 --model yolov8s.pt --conf 0.25 --max_frames 400
```

Writes:
- `assets/yolo_training_0010.jsonl`

### C) YOLO → tracker video

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/run_tracker_seq.py \
  --source yolo --split training --seq 0010 \
  --max_frames 250 --fps 10 \
  --car_conf 0.70 --ped_conf 0.30 \
  --min_h_car 30 --min_h_ped 20 \
  --iou_assoc_thresh 0.10 --max_age 30 --min_hits 2 \
  --out assets/demo_yolo_sort_0010.mp4
```

---

## Scripts and how to use them

Below are the main scripts you’ll use most often. (See `kitti-mot-kalman/scripts/` for the full list.)

### 1) `cache_yolo_dets.py` — cache YOLO detections to JSONL
Caches detections so you can tune thresholds without rerunning YOLO each time.

```bash
python kitti-mot-kalman/scripts/cache_yolo_dets.py \
  --split training --seq 0010 --model yolov8s.pt --conf 0.25 --max_frames 400
```

Outputs (example):
- `assets/yolo_training_0010.jsonl`

### 2) `run_tracker_seq.py` — render a tracker video for one sequence
Supports:
- `--source gt` (GT boxes as detections)
- `--source yolo` (cached YOLO detections)

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/run_tracker_seq.py --help
```

### 3) `eval_seq.py` — evaluate tracking using GT-as-detections
This is the best **sanity evaluation** for your association + lifecycle logic.

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/eval_seq.py --seq 0010 --drop 0 --jitter 0 --max_frames 300
```

### 4) `eval_yolo_tracker_seq.py` — evaluate YOLO → tracker on one sequence

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/eval_yolo_tracker_seq.py --seq 0010 --max_frames 300
```

### 5) `eval_yolo_all.py` — evaluate YOLO → tracker on all training sequences
Runs all 21 sequences and saves a CSV.

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/eval_yolo_all.py \
  --conf 0.25 --car_conf 0.70 --ped_conf 0.30 \
  --min_h_car 30 --min_h_ped 20 \
  --iou_assoc_thresh 0.10 --max_age 30 --min_hits 2 \
  --max_frames 400
```

Outputs:
- `assets/eval_yolo_all.csv`

### 6) `det_confmat_seq.py` — detection quality diagnostic (TP/FP/FN + confusion)
Useful for understanding whether tracking problems are coming from the detector.

```bash
PYTHONPATH=kitti-mot-kalman/src \
python kitti-mot-kalman/scripts/det_confmat_seq.py --seq 0010 --max_frames 400 --conf 0.6 --iou 0.5
```

---

## Tuning guide

A practical tuning workflow that keeps iteration fast:

1) **Cache YOLO once with a low threshold** (e.g. `--conf 0.25`).
2) Tune **runtime filters** and **tracker association/lifecycle**:

**Detection-side knobs**
- `car_conf`, `ped_conf`: raise to reduce FP, lower to reduce FN (more recalls)
- `min_h_car`, `min_h_ped`: filters tiny boxes that are usually noisy

**Tracker-side knobs**
- `iou_assoc_thresh`: higher → stricter matching (fewer wrong matches, more misses)
- `max_age`: higher → tracks survive longer during occlusion (but may drift)
- `min_hits`: higher → suppress short-lived false tracks (but delays track start)

Recommended approach:
- Start with a small dev set (e.g. `0003, 0007, 0010`) and run quick sweeps.
- Once stable, run `eval_yolo_all.py` to report the final mean metrics.

---

## Acknowledgements

- KITTI Tracking benchmark (dataset + labels)
- SORT baseline idea: Kalman filtering + Hungarian assignment on IoU

