import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from bedmonitor.config import GT_DIR
from bedmonitor.render import draw_frame
from bedmonitor.states import STATE_ORDER, UNKNOWN
from bedmonitor.temporal import detect_events, mmss, segments


def load_ground_truth(stem: str):
    path = GT_DIR / f"{stem}.csv"
    if not path.exists():
        return None
    gt = pd.read_csv(path)
    gt["state"] = gt["state"].str.strip().str.upper()
    return gt.sort_values("start_sec").reset_index(drop=True)


def gt_at(t, gt):
    match = gt[(gt["start_sec"] <= t) & (t < gt["end_sec"])]
    return match.iloc[0]["state"] if len(match) else None


def match_events(pred, truth, tol):
    used, tp = set(), 0
    for g in truth:
        for k, p in enumerate(pred):
            if k not in used and p["event"] == g["event"] and abs(p["start_sec"] - g["start_sec"]) <= tol:
                used.add(k)
                tp += 1
                break
    return tp, [p for k, p in enumerate(pred) if k not in used]


def evaluate(stem, df, timeline, events, times, duration, samples, bed_poly, out_dir, s):
    gt = load_ground_truth(stem)
    if gt is None:
        return None
    df["gt"] = [gt_at(t, gt) for t in df["t"]]
    ev = df.dropna(subset=["gt"])
    if ev.empty:
        return None

    ablation = [{"stage": name, "accuracy": round(accuracy_score(ev["gt"], ev[col]), 3),
                 "unknown_frames": int((ev[col] == UNKNOWN).sum())}
                for name, col in [("raw rules", "raw_state"), ("+ smoothing", "smoothed"),
                                  ("+ agent (VLM)", "agent_state"), ("+ state machine", "final")]]

    labels = [x for x in STATE_ORDER if x in set(ev["gt"]) | set(ev["final"])]
    ConfusionMatrixDisplay.from_predictions(ev["gt"], ev["final"], labels=labels, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=110)
    plt.close("all")

    gt_dur = {}
    for _, r in gt.iterrows():
        gt_dur[r["state"]] = gt_dur.get(r["state"], 0.0) + max(0.0, min(r["end_sec"], duration) - r["start_sec"])
    pred_dur = {}
    for seg in timeline:
        pred_dur[seg["state"]] = pred_dur.get(seg["state"], 0.0) + seg["duration_sec"]
    durations = [{"state": st, "ground_truth_sec": round(gt_dur.get(st, 0.0), 1),
                  "predicted_sec": round(pred_dur.get(st, 0.0), 1),
                  "error_sec": round(abs(gt_dur.get(st, 0.0) - pred_dur.get(st, 0.0)), 1)}
                 for st in STATE_ORDER if gt_dur.get(st, 0.0) > 0 or pred_dur.get(st, 0.0) > 0]

    gt_states = [x if isinstance(x, str) else UNKNOWN for x in df["gt"]]
    gt_events, _ = detect_events(gt_states, times, [1.0] * len(times), s)
    event_metrics = {}
    for kind in ("bed_exit", "return_to_bed"):
        p = [e for e in events if e["event"] == kind]
        g = [e for e in gt_events if e["event"] == kind]
        tp, fp = match_events(p, g, s.event_tolerance_sec)
        event_metrics[kind] = {"ground_truth": len(g), "predicted": len(p), "true_positives": tp,
                               "precision": round(tp / len(p), 3) if p else None,
                               "recall": round(tp / len(g), 3) if g else None,
                               "false_detections_at": [e["start_time"] for e in fp]}

    wrong = [isinstance(g, str) and f != g for f, g in zip(df["final"], df["gt"])]
    error_segs = sorted([(a, b) for flag, a, b in segments(wrong) if flag and b - a >= 2],
                        key=lambda x: x[1] - x[0], reverse=True)[:3]
    failures = []
    for n, (a, b) in enumerate(error_segs, 1):
        mid = (a + b) // 2
        row = df.iloc[mid]
        subject = None if row["subject"] is None or row["subject"] != row["subject"] else int(row["subject"])
        img = draw_frame(samples[mid], subject, bed_poly, s.kp_th,
                         f"{mmss(row['t'])} truth={row['gt']} pred={row['final']}", (0, 0, 255))
        name = f"failure_{n}.png"
        cv2.imwrite(str(out_dir / name), img)
        failures.append({"image": name, "from": mmss(df["t"].iloc[a]), "to": mmss(df["t"].iloc[b - 1]),
                         "truth": row["gt"], "predicted": row["final"],
                         "features": {"people": int(row["people"]), "visible_kp": int(row["visible_kp"]),
                                      "torso_angle": None if pd.isna(row["torso_angle"]) else round(row["torso_angle"]),
                                      "knee_angle": None if pd.isna(row["knee_angle"]) else round(row["knee_angle"]),
                                      "hip_in_bed": None if pd.isna(row["hip_in_bed"]) else bool(row["hip_in_bed"])},
                         "why": "TODO: explain the cause"})

    result = {
        "frames_evaluated": int(len(ev)),
        "accuracy": round(accuracy_score(ev["gt"], ev["final"]), 3),
        "ablation": ablation,
        "per_state": classification_report(ev["gt"], ev["final"], labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "durations": durations,
        "mean_duration_error_sec": round(sum(d["error_sec"] for d in durations) / max(1, len(durations)), 1),
        "events": event_metrics,
        "failure_cases": failures,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result
