import json

import cv2
import numpy as np
import pytest

import bedmonitor.evaluate as evaluate_mod
import bedmonitor.pipeline as pipeline
from bedmonitor.config import Settings
from bedmonitor.temporal import detect_events, enforce_transitions, smooth

BED = np.array([[100, 200], [500, 200], [500, 380], [100, 380]], dtype=np.int32)
SCRIPT = [(0, 20, "LYING_IN_BED"), (20, 28, "SITTING_ON_BED"), (28, 31, "STANDING"), (31, 45, "WALKING"),
          (45, 50, "OUT_OF_BED"), (50, 55, "WALKING"), (55, 62, "SITTING_ON_BED"), (62, 80, "LYING_IN_BED")]


def body(head, shoulders, hips, knee, ankle):
    pts = [head] * 5 + [shoulders[0], shoulders[1]] * 3 + [hips[0], hips[1], knee, knee, ankle, ankle]
    return np.array(pts, dtype=float)


def person(kxy, conf=0.9, track_id=1, box_conf=0.85):
    return {"track_id": track_id, "box": np.array([kxy[:, 0].min() - 10, kxy[:, 1].min() - 10, kxy[:, 0].max() + 10, kxy[:, 1].max() + 10]),
            "conf": box_conf, "kxy": kxy, "kconf": np.full(17, conf)}


LYING = body((140, 290), [(180, 280), (180, 300)], [(330, 285), (330, 305)], (400, 295), (460, 295))
EDGE = body((315, 170), [(300, 200), (330, 200)], [(305, 330), (325, 330)], (360, 340), (360, 430))


def standing(x):
    return body((x, 120), [(x - 15, 150), (x + 15, 150)], [(x - 10, 280), (x + 10, 280)], (x, 360), (x, 440))


def people_at(t):
    if t < 20:
        return [person(LYING, 0.2)] if 8 <= t < 10 else [person(LYING)]
    if t < 28:
        return [person(EDGE)]
    if t < 31:
        return [person(standing(575))]
    if t < 45 or 50 <= t < 55:
        return [person(standing(540 + abs((t * 100) % 180 - 90)))]
    if t < 50:
        return []
    if t < 62:
        return [person(EDGE)]
    return [person(LYING)]


def make_workspace(tmp_path, monkeypatch, people_fn, script, seconds=80, name="synthetic"):
    video = tmp_path / f"{name}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))
    for _ in range(8 * seconds):
        writer.write(np.full((480, 640, 3), 40, np.uint8))
    writer.release()
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir(exist_ok=True)
    (gt_dir / f"{name}.csv").write_text(
        "start_sec,end_sec,state\n" + "\n".join(f"{a},{b},{st}" for a, b, st in script) + "\n")

    def fake_pose(samples, *args, **kwargs):
        for s in samples:
            s["people"] = people_fn(s["t"])

    monkeypatch.setattr(pipeline, "run_pose", fake_pose)
    monkeypatch.setattr(pipeline, "get_bed_polygon", lambda *a, **k: (BED, "test"))
    monkeypatch.setattr(pipeline, "OUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(evaluate_mod, "GT_DIR", gt_dir)
    return video


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))
    for _ in range(8 * 80):
        writer.write(np.full((480, 640, 3), 40, np.uint8))
    writer.release()

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "synthetic.csv").write_text(
        "start_sec,end_sec,state\n" + "\n".join(f"{a},{b},{st}" for a, b, st in SCRIPT) + "\n")

    def fake_pose(samples, *args, **kwargs):
        for s in samples:
            s["people"] = people_at(s["t"])

    monkeypatch.setattr(pipeline, "run_pose", fake_pose)
    monkeypatch.setattr(pipeline, "get_bed_polygon", lambda *a, **k: (BED, "test"))
    monkeypatch.setattr(pipeline, "OUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(evaluate_mod, "GT_DIR", gt_dir)
    return video, tmp_path / "outputs" / "synthetic"


def test_end_to_end(workspace):
    video, out = workspace
    result = pipeline.analyze(video, use_vlm=False, render=True, settings=Settings(sample_fps=4))

    states = [seg["state"] for seg in result["timeline"]]
    assert states[0] == "LYING_IN_BED" and states[-1] == "LYING_IN_BED"
    assert [e["event"] for e in result["events"]] == ["bed_exit", "return_to_bed"]
    total = sum(seg["duration_sec"] for seg in result["timeline"])
    assert total == pytest.approx(result["summary"]["observation_duration_sec"], abs=0.5)
    assert result["evaluation"]["accuracy"] > 0.9
    assert result["evaluation"]["events"]["bed_exit"]["recall"] == 1.0
    for name in ["timeline.txt", "summary.json", "events.json", "alerts.json", "agent_trace.json",
                 "evaluation.json", "confusion_matrix.png", "stages.png", "annotated.mp4", "frames.csv"]:
        assert (out / name).exists(), name
    json.loads((out / "summary.json").read_text())


def run_events(states, s=Settings()):
    times = [i * 0.25 for i in range(len(states))]
    return detect_events(states, times, [0.9] * len(states), s)


def test_sitting_up_is_not_a_bed_exit():
    events, _ = run_events(["LYING_IN_BED"] * 40 + ["SITTING_ON_BED"] * 40 + ["LYING_IN_BED"] * 40)
    assert events == []


def test_brief_stand_is_cancelled():
    events, cancelled = run_events(["SITTING_ON_BED"] * 24 + ["STANDING"] * 6 + ["SITTING_ON_BED"] * 40)
    assert events == [] and [c["event"] for c in cancelled] == ["bed_exit"]


def test_smoothing_removes_flicker():
    states = ["LYING_IN_BED"] * 20 + ["SITTING_ON_BED"] + ["LYING_IN_BED"] * 20
    assert set(smooth(states, 0.25, Settings())) == {"LYING_IN_BED"}


def test_illegal_jump_is_held():
    states = ["LYING_IN_BED"] * 10 + ["WALKING"] * 2 + ["LYING_IN_BED"] * 10
    out, flags = enforce_transitions(states, [i * 0.25 for i in range(len(states))], hold=4)
    assert set(out) == {"LYING_IN_BED"} and flags == []


def test_vlm_disabled_without_settings(monkeypatch):
    from bedmonitor import vlm
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "OPENAI_BASE_URL", "VLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert vlm.make_vlm() is None


def test_vlm_parses_bedrock_reply(monkeypatch):
    from types import SimpleNamespace
    from bedmonitor import vlm
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("VLM_MODEL", "test-model")
    client = vlm.make_vlm()
    reply = '```json\n{"state": "LYING_IN_BED", "location": "bed", "confidence": 0.9, "reasoning": "on the mattress"}\n```'
    fake = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kw: fake)
    answer = client.ask([np.zeros((100, 100, 3), np.uint8)], "What state?")
    assert answer.state == "LYING_IN_BED" and answer.location == "bed"


def caregiver_scene(t):
    patient = people_at(t)
    if 15 <= t < 35:
        caregiver = person(standing(60), track_id=2, box_conf=0.97)
        if 22 <= t < 25:
            return [caregiver]
        return patient + [caregiver]
    return patient


def test_caregiver_is_ignored_and_exit_is_assisted(tmp_path, monkeypatch):
    video = make_workspace(tmp_path, monkeypatch, caregiver_scene, SCRIPT, name="caregiver")
    result = pipeline.analyze(video, use_vlm=False, render=False, settings=Settings(sample_fps=4))
    assert result["meta"]["patient_track"] == 1
    exits = [e for e in result["events"] if e["event"] == "bed_exit"]
    assert len(exits) == 1 and exits[0]["assisted"] and exits[0]["decision"] == "NORMAL"
    states_22_25 = {seg["state"] for seg in result["timeline"] if seg["start_sec"] < 25 and seg["end_sec"] > 22}
    assert states_22_25 == {"SITTING_ON_BED"}
    assert result["evaluation"]["accuracy"] > 0.9


BLANKET_SCRIPT = [(0, 120, "LYING_IN_BED")]


def blanket_scene(t):
    if 40 <= t < 100:
        return []
    if 30 <= t < 40 or 100 <= t < 110:
        return [person(LYING, 0.2)]
    return [person(LYING)]


def test_blanket_keeps_person_in_bed(tmp_path, monkeypatch):
    video = make_workspace(tmp_path, monkeypatch, blanket_scene, BLANKET_SCRIPT, seconds=120, name="blanket")
    result = pipeline.analyze(video, use_vlm=False, render=False, settings=Settings(sample_fps=4))
    assert [seg["state"] for seg in result["timeline"]] == ["LYING_IN_BED"]
    assert result["events"] == [] and result["overall_decision"] == "NORMAL"
    assert any("blanket" in step["observation"] for step in result["agent_trace"])
