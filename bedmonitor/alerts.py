from bedmonitor.states import LYING_OUTSIDE_BED, OUT_OF_BED, SITTING_ON_BED, UNKNOWN
from bedmonitor.temporal import hms

LEVEL = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


def alert(level, t, rule, detail):
    return {"level": level, "time": hms(t), "time_sec": round(t, 2), "rule": rule, "detail": detail}


def evaluate_alerts(timeline, out_periods, events, s):
    alerts = []
    for seg in timeline:
        d, t0, st = seg["duration_sec"], seg["start_sec"], seg["state"]
        if st == SITTING_ON_BED and d >= s.sit_edge_monitor_sec:
            alerts.append(alert("MONITOR", t0 + s.sit_edge_monitor_sec, "prolonged_sitting_on_bed",
                                f"Sitting on bed for {d:.0f}s"))
        if st == UNKNOWN and d >= s.unknown_monitor_sec:
            alerts.append(alert("MONITOR", t0 + s.unknown_monitor_sec, "prolonged_unknown",
                                f"State could not be determined for {d:.0f}s"))
        if st == OUT_OF_BED and d >= s.out_of_view_monitor_sec:
            alerts.append(alert("MONITOR", t0 + s.out_of_view_monitor_sec, "out_of_view",
                                f"Person out of camera view for {d:.0f}s"))
        if st == LYING_OUTSIDE_BED and d >= s.lying_outside_alert_sec:
            alerts.append(alert("ALERT", t0 + s.lying_outside_alert_sec, "lying_outside_bed",
                                f"Lying outside bed for {d:.0f}s, possible fall"))
    for a, b in out_periods:
        if b - a >= s.out_of_bed_alert_sec:
            alerts.append(alert("ALERT", a + s.out_of_bed_alert_sec, "prolonged_absence_from_bed",
                                f"Out of bed for {b - a:.0f}s"))
    for e in events:
        e["decision"] = "MONITOR" if e["event"] == "bed_exit" and not e.get("assisted") else "NORMAL"
        if e["event"] == "bed_exit" and e["current_state"] == LYING_OUTSIDE_BED.lower():
            e["decision"] = "ALERT"
    overall = max([a["level"] for a in alerts] + [e["decision"] for e in events], key=LEVEL.get, default="NORMAL")
    return sorted(alerts, key=lambda x: x["time_sec"]), overall
