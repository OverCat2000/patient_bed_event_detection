# Data

| Folder | Contents |
|---|---|
| `videos/` | Input videos (git-ignored). The file name without extension is the video id. |
| `ground_truth/` | `<video id>.csv` with hand-labeled segments. Copy `_template.csv` to start. |
| `bed_regions/` | Optional `<video id>.json` bed polygon, created with `python -m bedmonitor.bed_region <video>`. |

Ground-truth rows are `start_sec,end_sec,state`, covering the whole video without gaps, using these states:
`LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, LYING_OUTSIDE_BED, UNKNOWN`.
Label the monitored person only, from the video itself, never from the system's output.
