import json

from bedmonitor.config import OUT_DIR, ROOT


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def table(rows, cols):
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    runs = sorted(d for d in OUT_DIR.iterdir() if d.is_dir() and (d / "summary.json").exists())
    out = ["# Results report\n", "Generated from `outputs/`. Rerun `python -m bedmonitor.report` after new runs.\n"]

    overview = []
    for d in runs:
        summary = load(d / "summary.json")["summary"]
        ev = load(d / "evaluation.json")
        overview.append({
            "video": d.name,
            "duration_sec": summary["observation_duration_sec"],
            "bed_exits": summary["bed_exit_count"],
            "bed_returns": summary["bed_return_count"],
            "decision": summary.get("overall_decision"),
            "accuracy": ev["accuracy"] if ev else "no ground truth",
            "mean_duration_error_sec": ev["mean_duration_error_sec"] if ev else "",
            "bed_exit_precision": ev["events"]["bed_exit"]["precision"] if ev else "",
            "bed_exit_recall": ev["events"]["bed_exit"]["recall"] if ev else "",
        })
    out.append("## Overview\n")
    out.append(table(overview, ["video", "duration_sec", "bed_exits", "bed_returns", "decision", "accuracy",
                                "mean_duration_error_sec", "bed_exit_precision", "bed_exit_recall"]))

    for d in runs:
        rel = d.relative_to(ROOT)
        summary = load(d / "summary.json")
        events = load(d / "events.json")
        alerts = load(d / "alerts.json")
        ev = load(d / "evaluation.json")
        out.append(f"\n## {d.name}\n")
        out.append("### Timeline\n```\n" + (d / "timeline.txt").read_text() + "```\n")
        out.append("### Activity summary\n```json\n" + json.dumps(summary["readable"], indent=2) + "\n```\n")
        out.append("```json\n" + json.dumps(summary["summary"], indent=2) + "\n```\n")
        out.append("### Bed events\n")
        out.append(table(events["events"], ["event", "start_time", "confirmed_time", "previous_state",
                                            "current_state", "confidence", "assisted", "decision"]))
        out.append(f"\n### Alerts (overall: {alerts['overall_decision']})\n")
        out.append(table(alerts["alerts"], ["level", "time", "rule", "detail"]))
        out.append(f"\n![stages]({rel}/stages.png)\n")
        if ev:
            out.append("\n### Evaluation\n")
            out.append(f"Frame accuracy: **{ev['accuracy']}** over {ev['frames_evaluated']} sampled frames.\n\n")
            out.append(table(ev["ablation"], ["stage", "accuracy", "unknown_frames"]))
            out.append("\nDuration error\n\n")
            out.append(table(ev["durations"], ["state", "ground_truth_sec", "predicted_sec", "error_sec"]))
            out.append("\nBed events\n\n")
            out.append(table([{"event": k, **v} for k, v in ev["events"].items()],
                             ["event", "ground_truth", "predicted", "true_positives", "precision", "recall",
                              "false_detections_at"]))
            out.append(f"\n![confusion matrix]({rel}/confusion_matrix.png)\n")
            if ev["failure_cases"]:
                out.append("\n### Failure cases\n")
                for f in ev["failure_cases"]:
                    out.append(f"\n**{f['from']}–{f['to']}**: truth `{f['truth']}`, predicted `{f['predicted']}`. "
                               f"Features: {f['features']}. Why: {f['why']}\n\n![failure]({rel}/{f['image']})\n")

    path = ROOT / "REPORT.md"
    path.write_text("\n".join(out))
    print(f"Wrote {path} ({len(runs)} videos)")


if __name__ == "__main__":
    main()
