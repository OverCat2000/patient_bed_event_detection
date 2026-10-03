from pathlib import Path

import cv2


def sample_video(path: Path, sample_fps: float, max_width: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / sample_fps))
    scale = min(1.0, max_width / width) if width else 1.0

    samples, idx = [], 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            samples.append({"frame_idx": idx, "t": idx / fps, "image": frame})
        idx += 1
    cap.release()

    n_frames = n_frames or idx
    meta = {"video": path.name, "fps": fps, "frame_count": n_frames, "width": width, "height": height,
            "duration_sec": n_frames / fps, "sample_fps": sample_fps, "step": step,
            "sample_interval_sec": step / fps, "scale": scale, "samples": len(samples)}
    return samples, meta
