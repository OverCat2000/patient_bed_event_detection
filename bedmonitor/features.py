import cv2
import numpy as np
import pandas as pd

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, L_ANKLE = 5, 6, 11, 12, 13, 15


def kp(xy, conf, i, th):
    return xy[i] if conf[i] >= th else None


def midpoint(xy, conf, a, b, th):
    pa, pb = kp(xy, conf, a, th), kp(xy, conf, b, th)
    if pa is not None and pb is not None:
        return (pa + pb) / 2
    return pa if pa is not None else pb


def torso_angle(shoulders, hips):
    dx, dy = hips - shoulders
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))


def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1, 1))))


def in_bed(point, poly):
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def box_overlap(box, poly):
    bx1, by1, bx2, by2 = box
    px1, py1 = poly[:, 0].min(), poly[:, 1].min()
    px2, py2 = poly[:, 0].max(), poly[:, 1].max()
    inter = max(0.0, min(bx2, px2) - max(bx1, px1)) * max(0.0, min(by2, py2) - max(by1, py1))
    area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / area)


def build_features(samples, bed_poly, th: float) -> pd.DataFrame:
    rows = []
    for s in samples:
        people = s.get("people", [])
        i = s.get("subject")
        if i is None and "subject" not in s and people:
            i = int(np.argmax([p["conf"] for p in people]))
        others = len(people) - (1 if i is not None else 0)
        row = {"t": s["t"], "frame_idx": s["frame_idx"], "people": len(people), "subject": i,
               "patient_found": i is not None, "caregiver_present": others >= 1}
        if i is not None:
            p = people[i]
            xy, c = p["kxy"], p["kconf"]
            sh, hp = midpoint(xy, c, L_SHOULDER, R_SHOULDER, th), midpoint(xy, c, L_HIP, R_HIP, th)
            kh, kk, ka = kp(xy, c, L_HIP, th), kp(xy, c, L_KNEE, th), kp(xy, c, L_ANKLE, th)
            row.update(
                track_id=p.get("track_id"),
                box_conf=p["conf"],
                visible_kp=int((c >= th).sum()),
                torso_angle=torso_angle(sh, hp) if sh is not None and hp is not None else np.nan,
                knee_angle=joint_angle(kh, kk, ka) if kh is not None and kk is not None and ka is not None else np.nan,
                hip_in_bed=in_bed(hp, bed_poly) if hp is not None else np.nan,
                bed_overlap=box_overlap(p["box"], bed_poly),
                hip_x=hp[0] if hp is not None else np.nan,
                hip_y=hp[1] if hp is not None else np.nan,
                box_h=float(p["box"][3] - p["box"][1]),
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["track_id", "box_conf", "visible_kp", "torso_angle", "knee_angle", "hip_in_bed", "bed_overlap",
                "hip_x", "hip_y", "box_h"]:
        if col not in df:
            df[col] = np.nan
    df["visible_kp"] = df["visible_kp"].fillna(0)
    dist = np.hypot(df["hip_x"].diff(), df["hip_y"].diff())
    df["speed"] = (dist / df["t"].diff() / df["box_h"]).rolling(3, min_periods=1).mean()
    return df
