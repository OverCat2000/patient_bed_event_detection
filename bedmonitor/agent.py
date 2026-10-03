from collections import Counter

import cv2

from bedmonitor.states import IN_STATES, LYING_IN_BED, LYING_OUTSIDE_BED, OUT_OF_BED, OUT_STATES, STATE_ORDER, \
    UNKNOWN, WALKING
from bedmonitor.temporal import mmss, segments


class Agent:
    def __init__(self, samples, times, duration, dt, vlm, s, reasons=None, caregiver=None):
        self.samples, self.times, self.duration = samples, times, duration
        self.reasons = reasons or [""] * len(times)
        self.caregiver = caregiver or [False] * len(times)
        self.vlm, self.s = vlm, s
        self.context = max(1, round(s.context_sec / dt))
        self.trace, self.vlm_calls = [], 0

    def log(self, t, observation, thought, action, finding, conclusion):
        self.trace.append({"time": mmss(t), "observation": observation, "thought": thought,
                           "action": action, "finding": finding, "conclusion": conclusion})

    def end(self, b):
        return self.times[b] if b < len(self.times) else self.duration

    def look_back(self, states, idx):
        known = [x for x in states[max(0, idx - self.context): idx] if x != UNKNOWN]
        return known[-1] if known else None

    def look_ahead(self, states, idx):
        known = [x for x in states[idx: idx + self.context] if x != UNKNOWN]
        return known[0] if known else None

    def frame_at(self, t):
        sample = min(self.samples, key=lambda x: abs(x["t"] - t))
        img = sample["image"].copy()
        i = sample.get("subject")
        if i is not None:
            x1, y1, x2, y2 = map(int, sample["people"][i]["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return img

    def ask_vlm(self, t, question):
        if self.vlm is None or self.vlm_calls >= self.s.max_vlm_calls:
            return None
        self.vlm_calls += 1
        return self.vlm.ask([self.frame_at(x) for x in (t - 1, t, t + 1)], question)

    def resolve(self, states):
        states = list(states)
        for st, a, b in segments(states):
            if st != UNKNOWN:
                continue
            t0, t1 = self.times[a], self.end(b)
            before, after = self.look_back(states, a), self.look_ahead(states, b)
            if before is not None and before == after and t1 - t0 <= self.s.fill_max_sec:
                states[a:b] = [before] * (b - a)
                self.log(t0, f"UNKNOWN for {t1 - t0:.1f}s", "Current frames insufficient; check surrounding context.",
                         "look_back + look_ahead", f"{before} before and after", f"Relabel as {before}")
            else:
                self.log(t0, f"UNKNOWN for {t1 - t0:.1f}s", "Current frames insufficient; check surrounding context.",
                         "look_back + look_ahead", f"before={before}, after={after}", "Unresolved, escalate to VLM")

        for st, a, b in segments(states):
            if st not in (UNKNOWN, LYING_OUTSIDE_BED):
                continue
            t0 = self.times[a]
            mid = (t0 + self.end(b)) / 2
            if st == UNKNOWN:
                ans = self.ask_vlm(mid, "What state is the person in? Are they on the bed, floor, or elsewhere?")
                if ans and ans.state in STATE_ORDER and ans.state != UNKNOWN and ans.confidence >= self.s.vlm_min_conf:
                    states[a:b] = [ans.state] * (b - a)
                    self.log(t0, "UNKNOWN after context check", "Ask the VLM to inspect the frames.", "ask_vlm",
                             f"{ans.state} on {ans.location} ({ans.confidence:.2f}): {ans.reasoning}",
                             f"Relabel as {ans.state}")
                else:
                    self.log(t0, "UNKNOWN after context check", "Ask the VLM to inspect the frames.", "ask_vlm",
                             "VLM unavailable or not confident", "Keep UNKNOWN")
            else:
                ans = self.ask_vlm(mid, "Is the person lying on the bed or on the floor?")
                if ans and ans.location == "bed" and ans.confidence >= self.s.vlm_min_conf:
                    states[a:b] = [LYING_IN_BED] * (b - a)
                    self.log(t0, "Lying outside bed region", "Determine bed or floor.", "ask_vlm",
                             f"On bed ({ans.confidence:.2f}): {ans.reasoning}",
                             "Bed region too small; relabel LYING_IN_BED, decision NORMAL")
                elif ans and ans.location == "floor":
                    self.log(t0, "Lying outside bed region", "Determine bed or floor.", "ask_vlm",
                             f"On floor ({ans.confidence:.2f}): {ans.reasoning}", "Possible fall confirmed, ALERT")
                else:
                    self.log(t0, "Lying outside bed region", "Determine bed or floor.", "ask_vlm",
                             "VLM unavailable or inconclusive", "Keep as possible fall, ALERT rule applies")
        return states

    def verify_inferred(self, states):
        states = list(states)
        flags = [r in ("covered_in_bed", "occluded_in_bed") for r in self.reasons]
        for flag, a, b in segments(flags):
            if not flag:
                continue
            t0, t1 = self.times[a], self.end(b)
            kind = "blanket or occlusion" if self.reasons[a] == "covered_in_bed" else "a caregiver in front"
            observation = f"Patient not clearly visible for {t1 - t0:.1f}s, likely {kind}; kept as {states[a]}"
            if t1 - t0 < self.s.covered_verify_sec:
                self.log(t0, observation, "Short gap after an in-bed state; no exit was seen.", "persist_last_state",
                         "No movement out of the bed region", f"Keep {states[a]}")
                continue
            ans = self.ask_vlm((t0 + t1) / 2, "Is a person lying or sitting in this bed, possibly under a blanket?")
            if ans and ans.location != "bed" and ans.confidence >= self.s.vlm_min_conf:
                for i in range(a, b):
                    if flags[i]:
                        states[i] = UNKNOWN
                self.log(t0, observation, "Long hidden period; confirm the person is still in bed.", "ask_vlm",
                         f"{ans.location} ({ans.confidence:.2f}): {ans.reasoning}",
                         "Not confirmed in bed; relabel UNKNOWN so the monitoring rule applies")
            else:
                finding = (f"{ans.location} ({ans.confidence:.2f}): {ans.reasoning}" if ans
                           else "VLM unavailable or not confident")
                self.log(t0, observation, "Long hidden period; confirm the person is still in bed.", "ask_vlm",
                         finding, f"Keep {states[a]}")
        return states

    def note_caregiver(self, subject_info):
        for present, a, b in segments(self.caregiver):
            if present and self.end(b) - self.times[a] >= 1.0:
                self.log(self.times[a], "Another person is in the scene",
                         "They may be a caregiver; keep following the monitored person.",
                         "track_patient", f"Patient is track {subject_info.get('patient_track')}",
                         "Use the patient's states only; assisted bed exits are NORMAL")
        for sw in subject_info.get("id_switches", []):
            self.log(sw["t"], "Patient's track was lost", "The tracker may have reassigned the ID.",
                     "match_by_position", f"New track {sw['to']} appeared where the patient was",
                     f"Continue with track {sw['to']}")

    def verify_events(self, events, states):
        for e in events:
            a, b = e["start_idx"], e["confirmed_idx"]
            before = dict(Counter(states[max(0, a - self.context): a]))
            after = dict(Counter(states[a: b + self.context]))
            if e["event"] == "bed_exit":
                prior_in_bed = any(x in IN_STATES for x in before)
                moved_away = any(x in (WALKING, OUT_OF_BED) for x in after)
                if prior_in_bed and moved_away:
                    e["confidence"] = min(0.99, round(e["confidence"] + 0.05, 2))
                    conclusion = "BED_EXIT confirmed: was in bed, then moved away"
                elif prior_in_bed:
                    conclusion = "BED_EXIT confirmed by persistence outside the bed"
                else:
                    e["confidence"] = round(e["confidence"] * 0.8, 2)
                    conclusion = "BED_EXIT weak: no clear in-bed state beforehand"
                self.log(e["start_sec"], "Person left the bed region",
                         "One frame is insufficient to confirm a bed exit; check before and after.",
                         "look_back + look_ahead", f"before={before}, after={after}", conclusion)
            else:
                prior_out = any(x in OUT_STATES for x in before)
                settled = any(x in IN_STATES for x in after)
                self.log(e["start_sec"], "Person entered the bed region",
                         "Confirm the person was out of bed and settled back in bed.",
                         "look_back + look_ahead", f"before={before}, after={after}",
                         "RETURN_TO_BED confirmed" if prior_out and settled else "RETURN_TO_BED weak")
        return events
