import math
from collections import Counter

import numpy as np

from bedmonitor.states import ALLOWED, IN_STATES, OUT_STATES, UNKNOWN


def mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def human(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}m {sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def segments(states):
    segs, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            segs.append([states[start], start, i])
            start = i
    return segs


def majority_filter(states, window: int):
    half, out = window // 2, []
    for i in range(len(states)):
        counts = Counter(states[max(0, i - half): i + half + 1]).most_common()
        top = {x for x, n in counts if n == counts[0][1]}
        out.append(states[i] if states[i] in top else counts[0][0])
    return out


def min_duration(states, min_len: int):
    states = list(states)
    while True:
        segs = segments(states)
        short = [k for k, (_, a, b) in enumerate(segs) if b - a < min_len]
        if len(segs) <= 1 or not short:
            return states
        k = short[0]
        neighbors = [segs[j] for j in (k - 1, k + 1) if 0 <= j < len(segs)]
        target = max(neighbors, key=lambda x: x[2] - x[1])[0]
        _, a, b = segs[k]
        states[a:b] = [target] * (b - a)


def smooth(states, dt: float, s):
    window = max(1, round(s.smooth_window_sec / dt)) | 1
    min_len = max(1, math.ceil(s.min_state_sec / dt))
    return min_duration(majority_filter(list(states), window), min_len)


def is_allowed(a, b) -> bool:
    return a == b or UNKNOWN in (a, b) or b in ALLOWED.get(a, set())


def enforce_transitions(states, times, hold: int):
    out, flags = [], []
    current, pending, count = None, None, 0
    for i, x in enumerate(states):
        if current is None or x == current:
            current = current or x
            pending, count = None, 0
        elif is_allowed(current, x):
            current, pending, count = x, None, 0
        else:
            count = count + 1 if x == pending else 1
            pending = x
            if count > hold:
                start = i - hold
                flags.append({"time": mmss(times[start]), "from": current, "to": x,
                              "note": "illegal transition accepted after persisting"})
                out[start:i] = [x] * hold
                current, pending, count = x, None, 0
        out.append(current)
    return out, flags


def bed_status(states):
    status, last = [], None
    for x in states:
        if x in IN_STATES:
            last = "in"
        elif x in OUT_STATES:
            last = "out"
        status.append(last)
    first = next((x for x in status if x), "in")
    return [x or first for x in status]


def detect_events(states, times, confs, s):
    status = bed_status(states)
    confirmed = status[0] if status else "in"
    events, cancelled, pending = [], [], None
    for i in range(1, len(states)):
        if pending and status[i] == confirmed:
            cancelled.append({"event": pending["event"], "start_time": hms(pending["start_sec"]),
                              "cancelled_time": hms(times[i]), "previous_state": pending["previous_state"]})
            pending = None
        if pending is None and status[i] != confirmed and status[i] != status[i - 1]:
            kind = "bed_exit" if status[i] == "out" else "return_to_bed"
            pending = {"event": kind, "start_idx": i, "start_sec": times[i],
                       "previous_state": states[i - 1].lower()}
        if pending:
            need = s.exit_confirm_sec if pending["event"] == "bed_exit" else s.return_confirm_sec
            if times[i] - pending["start_sec"] >= need:
                window = [c for c in confs[pending["start_idx"]: i + 1] if c == c]
                events.append({
                    "event": pending["event"],
                    "start_time": hms(pending["start_sec"]),
                    "confirmed_time": hms(times[i]),
                    "previous_state": pending["previous_state"],
                    "current_state": states[i].lower(),
                    "confidence": round(float(np.mean(window)), 2) if window else 0.0,
                    "start_sec": pending["start_sec"],
                    "start_idx": pending["start_idx"],
                    "confirmed_idx": i,
                })
                confirmed = status[i]
                pending = None
    return events, cancelled


def build_timeline(states, times, duration):
    def end(b):
        return times[b] if b < len(times) else duration
    out = []
    for st, a, b in segments(states):
        start = 0.0 if a == 0 else times[a]
        out.append({"state": st, "start": mmss(start), "end": mmss(end(b)),
                    "start_sec": round(start, 2), "end_sec": round(end(b), 2),
                    "duration_sec": round(end(b) - start, 2)})
    return out


def out_of_bed_periods(states, times, duration):
    status = bed_status(states)
    return [(times[a], times[b] if b < len(times) else duration) for st, a, b in segments(status) if st == "out"]
