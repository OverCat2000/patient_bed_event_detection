import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bedmonitor.config import BED_DIR

MAX_DISPLAY = 1280


def main():
    ap = argparse.ArgumentParser(description="Click the bed corners on a frame and save them")
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=0.0, help="time in seconds of the frame to use")
    args = ap.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {video}")

    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY / max(h, w))
    display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x / scale), int(y / scale)])

    win = "Click bed corners | u: undo | s: save | q: quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = display.copy()
        pts = (np.array(points) * scale).astype(np.int32)
        for p in pts:
            cv2.circle(canvas, tuple(p), 5, (0, 0, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], len(pts) >= 3, (0, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("s") and len(points) >= 3:
            BED_DIR.mkdir(parents=True, exist_ok=True)
            out = BED_DIR / f"{video.stem}.json"
            out.write_text(json.dumps({"video": video.name, "polygon": points}, indent=2))
            print(f"Saved {out}")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
