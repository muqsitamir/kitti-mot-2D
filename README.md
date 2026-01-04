# KITTI MOT Kalman

A simple Kalman filter based tracker for KITTI MOT dataset.


```
for s in $(seq -w 10 20); do                                                        
  python kitti-mot-kalman/scripts/cache_yolo_dets.py \
    --split training --seq "00$s" --model yolov8n.pt --conf 0.25
done
```


```
PYTHONPATH=kitti-mot-kalman/src python kitti-mot-kalman/scripts/run_tracker_yolo_seq.py \
  --seq 0000 --split training --max_frames 400 --out assets/pred_yolo_0000.mp4
```

```
python kitti-mot-kalman/scripts/cache_yolo_dets.py --seq 0000 --split training --max_frames 400 --model yolov8n.pt --conf 0.25
```

```
PYTHONPATH=kitti-mot-kalman/src python kitti-mot-kalman/scripts/eval_seq.py --seq 0000 --drop 0 --jitter 0 --max_frames 300 --min_hits 1
```

```
PYTHONPATH=kitti-mot-kalman/src python kitti-mot-kalman/scripts/eval_yolo_tracker_seq.py \
  --seq 0003 --max_frames 300
```