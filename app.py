import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bedmonitor.config import OUT_DIR, VIDEO_DIR, VIDEO_EXTS
from bedmonitor.pipeline import analyze

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bed Monitor")
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")

INDEX = (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...), use_vlm: bool = Form(True)):
    name = Path(video.filename or "").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(VIDEO_EXTS)}")
    dest = VIDEO_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        result = await run_in_threadpool(analyze, dest, use_vlm)
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")
    return JSONResponse(json.loads(json.dumps(result, default=str)))


@app.get("/results/{stem}")
def results(stem: str):
    folder = OUT_DIR / Path(stem).name
    if not (folder / "summary.json").exists():
        raise HTTPException(404, "No results for this video")
    files = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}
    return files
