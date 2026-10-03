import json
import time
from pathlib import Path


from bedmonitor.agent import Agent
from bedmonitor.alerts import evaluate_alerts
from bedmonitor.config import OUT_DIR, Settings
from bedmonitor.evaluate import evaluate
from bedmonitor.features import build_features
from bedmonitor.perception import get_bed_polygon, run_pose
from bedmonitor.render import annotated_video, plot_stages
from bedmonitor.rules import classify_all
from bedmonitor.subject import assign_subject
from bedmonitor.temporal import (build_timeline, detect_events, enforce_transitions, human, out_of_bed_periods,
                                 smooth)
from bedmonitor.video import sample_video
from bedmonitor.vlm import make_vlm


def build_summary(timeline, periods, events, duration, final_state):
    activity = {}
    for seg in timeline:
        activity[seg["state"].lower()] = activity.get(seg["state"].lower(), 0.0) + seg["duration_sec"]
    out_sec = sum(b - a for a, b in periods)
    in_sec = duration - out_sec
    exits = sum(e["event"] == "bed_exit" for e in events)
    readable = {
        "total_observation_time": human(duration),
        "activity_summary": {k: human(v) for k, v in activity.items()},
        "bed_summary": {"time_in_bed": human(in_sec), "time_out_of_bed": human(out_sec), "bed_exit_count": exits},
    }
    summary = {
        "observation_duration_sec": round(duration, 1),
        "activity_duration_sec": {k: round(v, 1) for k, v in activity.items()},
        "bed_exit_count": exits,
        "bed_return_count": sum(e["event"] == "return_to_bed" for e in events),
        "total_in_bed_sec": round(in_sec, 1),
        "total_out_of_bed_sec": round(out_sec, 1),
        "longest_out_of_bed_period_sec": round(max([b - a for a, b in periods], default=0.0), 1),
        "final_state": final_state.lower(),
    }
    return readable, summary


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


def analyze(video_path, use_vlm=True, render=True, settings=None):
    s = settings or Settings()
    video_path = Path(video_path)
    out_dir = OUT_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    samples, meta = sample_video(video_path, s.sample_fps, s.max_width)
    if not samples:
        raise ValueError(f"No frames could be read from {video_path}")
    run_pose(samples, s.pose_model, s.track, s.tracker)
    bed_poly, bed_source = get_bed_polygon(samples, video_path, meta["scale"], s.det_model)
    width = samples[0]["image"].shape[1]
    subject_info = assign_subject(samples, bed_poly, s.kp_th, s.max_jump_frac * width, s.new_track_sec)

    df = build_features(samples, bed_poly, s.kp_th)
    df["raw_state"], df["reason"] = classify_all(df, s)
    times = df["t"].tolist()
    duration = meta["duration_sec"]
    dt = float(df["t"].diff().median()) if len(df) > 1 else meta["sample_interval_sec"]
    df["smoothed"] = smooth(df["raw_state"], dt, s)

    vlm = make_vlm() if use_vlm else None
    agent = Agent(samples, times, duration, dt, vlm, s, list(df["reason"]), list(df["caregiver_present"]))
    agent.note_caregiver(subject_info)
    df["agent_state"] = agent.verify_inferred(agent.resolve(df["smoothed"]))
    final, flags = enforce_transitions(list(df["agent_state"]), times, max(1, round(s.illegal_hold_sec / dt)))
    df["final"] = final

    timeline = build_timeline(final, times, duration)
    events, cancelled = detect_events(final, times, df["box_conf"].tolist(), s)
    events = agent.verify_events(events, final)
    for e in events:
        near = df[(df["t"] >= e["start_sec"] - s.caregiver_window_sec) & (df["t"] <= e["start_sec"] + s.caregiver_window_sec)]
        e["assisted"] = bool(near["caregiver_present"].any())
    periods = out_of_bed_periods(final, times, duration)
    alerts, overall = evaluate_alerts(timeline, periods, events, s)
    readable, summary = build_summary(timeline, periods, events, duration, final[-1])
    summary["overall_decision"] = overall

    public_events = [{k: e[k] for k in ("event", "start_time", "confirmed_time", "previous_state",
                                        "current_state", "confidence", "assisted", "decision")} for e in events]
    evaluation = evaluate(video_path.stem, df, timeline, events, times, duration, samples, bed_poly, out_dir, s)

    (out_dir / "timeline.txt").write_text(
        "\n".join(f"{seg['start']} - {seg['end']}  {seg['state']}" for seg in timeline) + "\n")
    write_json(out_dir / "timeline.json", timeline)
    write_json(out_dir / "summary.json", {"readable": readable, "summary": summary})
    write_json(out_dir / "events.json", {"events": public_events, "cancelled_candidates": cancelled})
    write_json(out_dir / "alerts.json", {"overall_decision": overall, "alerts": alerts, "transition_flags": flags})
    write_json(out_dir / "agent_trace.json", {"vlm": getattr(vlm, "name", None), "vlm_calls": agent.vlm_calls,
                                              "trace": agent.trace})
    meta.update(bed_source=bed_source, processing_sec=round(time.time() - started, 1),
                patient_track=subject_info["patient_track"], id_switches=subject_info["id_switches"],
                caregiver_frames=int(df["caregiver_present"].sum()),
                inferred_frames=int(df["reason"].isin(["covered_in_bed", "occluded_in_bed"]).sum()))
    write_json(out_dir / "meta.json", meta)
    df.drop(columns=["subject"]).to_csv(out_dir / "frames.csv", index=False)
    plot_stages(df, out_dir / "stages.png")
    if render:
        annotated_video(samples, df, bed_poly, alerts, s.sample_fps, s.kp_th, out_dir / "annotated.mp4")

    return {
        "video": video_path.name,
        "output_dir": str(out_dir),
        "meta": meta,
        "timeline": timeline,
        "readable_summary": readable,
        "summary": summary,
        "events": public_events,
        "alerts": alerts,
        "overall_decision": overall,
        "agent_trace": agent.trace,
        "vlm_calls": agent.vlm_calls,
        "evaluation": evaluation,
    }
