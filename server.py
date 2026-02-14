from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    video_dir = "static/videos"
    # Hole alle mp4s, sortiert nach Erstellungsdatum (neueste oben)
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    videos.sort(
        key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True
    )

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "videos": videos, "latest": videos[0] if videos else None},
    )


@app.get("/api/latest")
async def get_latest():
    video_dir = "static/videos"
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not videos:
        return {"latest": None}
    # Sortiere nach Zeitstempel
    videos.sort(
        key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True
    )
    return {"latest": videos[0]}
