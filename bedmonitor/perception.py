import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from bedmonitor.config import BED_DIR

BED_CLASS_ID = 59


@lru_cache(maxsize=4)
def load_model(name: str):
    from ultralytics import YOLO
    return YOLO(name)


def extract_people(result):
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return []
    n = len(result.boxes)
    kconf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.zeros((n, 17))
    ids = result.boxes.id.int().tolist() if result.boxes.id is not None else [None] * n
    return [
        {
            "track_id": ids[i],
            "box": result.boxes.xyxy[i].cpu().numpy(),
            "conf": float(result.boxes.conf[i]),
            "kxy": result.keypoints.xy[i].cpu().numpy(),
            "kconf": kconf[i],
        }
        for i in range(n)
    ]


def run_pose(samples, model_name: str, track: bool = True, tracker: str = "bytetrack.yaml"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    for s in samples:
        if track:
            result = model.track(s["image"], persist=True, tracker=tracker, verbose=False)[0]
        else:
            result = model(s["image"], verbose=False)[0]
        s["people"] = extract_people(result)


def detect_bed(samples, model_name: str):
    model = load_model(model_name)
    for s in samples[::max(1, len(samples) // 10)]:
        r = model(s["image"], classes=[BED_CLASS_ID], verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = boxes[int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())]
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return None


def manual_bed(video_path: Path, scale: float):
    path = BED_DIR / f"{video_path.stem}.json"
    if not path.exists():
        return None
    poly = np.array(json.loads(path.read_text())["polygon"], dtype=float) * scale
    return poly.astype(np.int32)


def get_bed_polygon(samples, video_path: Path, scale: float, model_name: str):
    poly = manual_bed(video_path, scale)
    if poly is not None:
        return poly, "manual"
    poly = detect_bed(samples, model_name)
    if poly is not None:
        return poly, "yolo"
    h, w = samples[0]["image"].shape[:2]
    fallback = np.array([[w * 0.2, h * 0.4], [w * 0.8, h * 0.4], [w * 0.8, h * 0.9], [w * 0.2, h * 0.9]])
    return fallback.astype(np.int32), "fallback"
