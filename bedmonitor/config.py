from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
GT_DIR = ROOT / "data" / "ground_truth"
BED_DIR = ROOT / "data" / "bed_regions"
OUT_DIR = ROOT / "outputs"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    sample_fps: float = 2.0
    max_width: int = 960
    pose_model: str = "yolo11n-pose.pt"
    det_model: str = "yolo11n.pt"
    kp_th: float = 0.5
    lying_max_deg: float = 35
    upright_min_deg: float = 55
    knee_bent_max_deg: float = 140
    walk_speed: float = 0.25
    min_visible_kp: int = 4
    smooth_window_sec: float = 1.5
    min_state_sec: float = 1.5
    illegal_hold_sec: float = 1.0
    exit_confirm_sec: float = 3.0
    return_confirm_sec: float = 3.0
    context_sec: float = 8.0
    fill_max_sec: float = 30.0
    vlm_min_conf: float = 0.6
    max_vlm_calls: int = 20
    sit_edge_monitor_sec: float = 120
    unknown_monitor_sec: float = 30
    out_of_view_monitor_sec: float = 60
    lying_outside_alert_sec: float = 5
    out_of_bed_alert_sec: float = 600
    event_tolerance_sec: float = 5.0
    track: bool = True
    tracker: str = "bytetrack.yaml"
    max_jump_frac: float = 0.15
    new_track_sec: float = 2.0
    covered_overlap_min: float = 0.5
    covered_verify_sec: float = 20.0
    caregiver_window_sec: float = 5.0
