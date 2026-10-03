import pandas as pd

from bedmonitor.states import (IN_STATES, LYING_IN_BED, LYING_OUTSIDE_BED, NO_PERSON, OUT_OF_BED, OUT_STATES,
                               SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, UNKNOWN, WALKING)

HIDDEN_REASONS = {"low_visibility", "no_hips"}
INFERRED_REASONS = {"covered_in_bed", "occluded_in_bed"}


def classify(row, s):
    if not row["patient_found"]:
        return NO_PERSON, "no_person"
    if row["visible_kp"] < s.min_visible_kp:
        return UNKNOWN, "low_visibility"
    angle, bed = row["torso_angle"], row["hip_in_bed"]
    if pd.isna(angle) or pd.isna(bed):
        return UNKNOWN, "no_hips"
    if s.lying_max_deg < angle < s.upright_min_deg:
        return UNKNOWN, "ambiguous_posture"
    if angle <= s.lying_max_deg:
        return (LYING_IN_BED, "rule") if bed else (LYING_OUTSIDE_BED, "rule")
    if bed:
        return SITTING_ON_BED, "rule"
    if not pd.isna(row["knee_angle"]) and row["knee_angle"] <= s.knee_bent_max_deg:
        return SITTING_OUTSIDE_BED, "rule"
    if not pd.isna(row["speed"]) and row["speed"] >= s.walk_speed:
        return WALKING, "rule"
    return STANDING, "rule"


def classify_all(df, s):
    last_known, states, reasons = None, [], []
    for _, row in df.iterrows():
        state, why = classify(row, s)
        hidden = state == NO_PERSON or why in HIDDEN_REASONS
        if hidden and last_known in IN_STATES:
            on_bed = state == NO_PERSON or (not pd.isna(row["bed_overlap"])
                                            and row["bed_overlap"] >= s.covered_overlap_min)
            if on_bed:
                state = last_known
                why = "occluded_in_bed" if row["caregiver_present"] else "covered_in_bed"
        elif state == NO_PERSON:
            if row["caregiver_present"]:
                state, why = UNKNOWN, "occluded_by_caregiver"
            elif last_known in OUT_STATES:
                state, why = OUT_OF_BED, "left_view"
            else:
                state, why = UNKNOWN, "not_visible"
        elif state != UNKNOWN:
            last_known = state
        states.append(state)
        reasons.append(why)
    return states, reasons
