import io
import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bedmonitor.config import GT_DIR, OUT_DIR, VIDEO_DIR, VIDEO_EXTS
from bedmonitor.pipeline import analyze
from bedmonitor.states import STATE_ORDER

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bed Monitor")
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")

INDEX = (Path(__file__).parent / "static" / "index.html").read_text()


def read_ground_truth(upload: UploadFile) -> bytes:
    raw = upload.file.read()
    try:
        gt = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Ground truth must be a CSV file")
    gt.columns = [str(c).strip() for c in gt.columns]
    if list(gt.columns) != ["start_sec", "end_sec", "state"]:
        raise HTTPException(400, "Ground truth columns must be: start_sec,end_sec,state")
    if gt.empty:
        raise HTTPException(400, "Ground truth file has no rows")
    if not all(pd.api.types.is_numeric_dtype(gt[c]) for c in ["start_sec", "end_sec"]):
        raise HTTPException(400, "start_sec and end_sec must be numbers")
    unknown = set(gt["state"].astype(str).str.strip().str.upper()) - set(STATE_ORDER)
    if unknown:
        raise HTTPException(400, f"Unknown states in ground truth: {sorted(unknown)}. Allowed: {STATE_ORDER}")
    return raw


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...), ground_truth: UploadFile | None = File(None),
                        use_vlm: bool = Form(True)):
    name = Path(video.filename or "").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(VIDEO_EXTS)}")
    stem = Path(name).stem
    gt_path = GT_DIR / f"{stem}.csv"

    gt_bytes = read_ground_truth(ground_truth) if ground_truth is not None and ground_truth.filename else None
    if gt_bytes is not None:
        gt_path.write_bytes(gt_bytes)
        gt_source = "uploaded"
    else:
        gt_source = "existing" if gt_path.exists() else "none"

    dest = VIDEO_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        result = await run_in_threadpool(analyze, dest, use_vlm)
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")
    result["ground_truth_source"] = gt_source
    result["ground_truth_file"] = f"data/ground_truth/{stem}.csv"
    return JSONResponse(json.loads(json.dumps(result, default=str)))


@app.get("/results/{stem}")
def results(stem: str):
    folder = OUT_DIR / Path(stem).name
    if not (folder / "summary.json").exists():
        raise HTTPException(404, "No results for this video")
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}
