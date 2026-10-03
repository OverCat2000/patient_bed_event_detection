import argparse
import json
from pathlib import Path

from bedmonitor.config import VIDEO_DIR, VIDEO_EXTS
from bedmonitor.pipeline import analyze


def main():
    ap = argparse.ArgumentParser(description="Analyze bed-monitoring videos")
    ap.add_argument("videos", nargs="*", help="video paths (default: every video in data/videos)")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    videos = [Path(v) for v in args.videos] or sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found. Put them in {VIDEO_DIR} or pass a path.")
    for video in videos:
        print(f"\n=== {video.name} ===")
        result = analyze(video, use_vlm=not args.no_vlm, render=not args.no_render)
        for seg in result["timeline"]:
            print(f"{seg['start']} - {seg['end']}  {seg['state']}")
        print(json.dumps(result["readable_summary"], indent=2))
        for e in result["events"]:
            print(f"{e['event']}: {e['start_time']} -> {e['confirmed_time']} ({e['decision']})")
        print(f"Overall decision: {result['overall_decision']} | VLM calls: {result['vlm_calls']}")
        if result["evaluation"]:
            print(f"Accuracy: {result['evaluation']['accuracy']} | "
                  f"mean duration error: {result['evaluation']['mean_duration_error_sec']}s")
        print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
