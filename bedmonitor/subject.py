import numpy as np

from bedmonitor.features import L_HIP, R_HIP, in_bed, midpoint


def center(person):
    x1, y1, x2, y2 = person["box"]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def find_patient_track(samples, bed_poly, th):
    scores = {}
    for s in samples:
        for p in s.get("people", []):
            tid = p.get("track_id")
            if tid is None:
                continue
            hips = midpoint(p["kxy"], p["kconf"], L_HIP, R_HIP, th)
            on_bed = in_bed(hips, bed_poly) if hips is not None else in_bed(center(p), bed_poly)
            scores[tid] = scores.get(tid, 0) + (3 if on_bed else 1)
    return (max(scores, key=scores.get) if scores else None), scores


def assign_subject(samples, bed_poly, th, max_jump, new_track_sec):
    patient, scores = find_patient_track(samples, bed_poly, th)
    first_seen = {}
    for s in samples:
        for p in s.get("people", []):
            if p.get("track_id") is not None:
                first_seen.setdefault(p["track_id"], s["t"])

    switches, last = [], None
    for s in samples:
        people = s.get("people", [])
        idx = None
        if patient is None:
            if people:
                idx = int(np.argmax([p["conf"] for p in people]))
        else:
            ids = [p.get("track_id") for p in people]
            if patient in ids:
                idx = ids.index(patient)
            elif last is not None and people:
                dists = [np.linalg.norm(center(p) - last) for p in people]
                j = int(np.argmin(dists))
                tid = people[j].get("track_id")
                is_new = tid is None or s["t"] - first_seen.get(tid, s["t"]) <= new_track_sec
                if dists[j] <= max_jump and is_new:
                    idx = j
                    if tid is not None:
                        switches.append({"t": s["t"], "from": patient, "to": tid})
                        patient = tid
        if idx is not None:
            last = center(people[idx])
        s["subject"] = idx
    return {"patient_track": patient, "track_scores": scores, "id_switches": switches}
