import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bedmonitor.alerts import LEVEL
from bedmonitor.states import STATE_ORDER
from bedmonitor.temporal import mmss

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
LEVEL_COLOR = {"NORMAL": (0, 200, 0), "MONITOR": (0, 200, 255), "ALERT": (0, 0, 255)}


def draw_frame(sample, subject, bed_poly, kp_th, text=None, color=(255, 255, 255)):
    img = sample["image"].copy()
    cv2.polylines(img, [bed_poly], True, (255, 0, 0), 2)
    for i, p in enumerate(sample.get("people", [])):
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if i == subject else (160, 160, 160), 2)
        if i == subject:
            xy, c = p["kxy"], p["kconf"]
            for a, b in SKELETON:
                if c[a] >= kp_th and c[b] >= kp_th:
                    cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (255, 255, 0), 2)
            for (x, y), cf in zip(xy, c):
                if cf >= kp_th:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    if text:
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return img


def plot_stages(df, path):
    cols = [("raw rules", "raw_state"), ("smoothed", "smoothed"), ("agent", "agent_state"), ("final", "final")]
    if "gt" in df and df["gt"].notna().any():
        cols.append(("ground truth", "gt"))
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.2 * len(cols)), sharex=True, squeeze=False)
    for ax, (name, col) in zip(axes[:, 0], cols):
        y = [STATE_ORDER.index(x) if x in STATE_ORDER else np.nan for x in df[col]]
        ax.step(df["t"], y, where="post")
        ax.scatter(df["t"], y, s=6)
        ax.set_yticks(range(len(STATE_ORDER)), STATE_ORDER, fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def level_at(t, alerts):
    level = "NORMAL"
    for a in alerts:
        if a["time_sec"] <= t and LEVEL[a["level"]] > LEVEL[level]:
            level = a["level"]
    return level


def annotated_video(samples, df, bed_poly, alerts, fps, kp_th, path):
    if not samples:
        return
    h, w = samples[0]["image"].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for sample, (_, row) in zip(samples, df.iterrows()):
        level = level_at(row["t"], alerts)
        subject = None if row["subject"] is None or row["subject"] != row["subject"] else int(row["subject"])
        img = draw_frame(sample, subject, bed_poly, kp_th, f"{mmss(row['t'])}  {row['final']}  [{level}]",
                         LEVEL_COLOR[level])
        writer.write(img)
    writer.release()
