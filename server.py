import os

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


def remove_file(path: str):
	try:
		if os.path.exists(path):
			os.remove(path)
			print(f'Datei gelöscht: {path}')
	except Exception as e:
		print(f'Fehler beim Löschen: {e}')


@app.get('/', response_class=HTMLResponse)
async def index(
	request: Request,
):
	video_dir = 'static/videos'
	videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
	videos.sort(key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)
	return templates.TemplateResponse(
		'index.html',
		{'request': request, 'video': videos[0] if videos else None},
	)


@app.get('/download/videos/{video_name}')
async def get_video(video_name: str, background_tasks: BackgroundTasks):
	file_path = f'static/videos/{video_name}'
	background_tasks.add_task(remove_file, file_path)
	return FileResponse(path=file_path, filename=video_name, media_type='video/mp4')
