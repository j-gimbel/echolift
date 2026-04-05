# read yaml config  files
import logging

# import yaml
import time
import tomllib
from enum import Enum

import cv2
import numpy as np
import qrcode

# from cv2 import config
from PIL.Image import Image
from pydantic import BaseModel
from qrcode.image.pil import PilImage


class Config(BaseModel):
	server_url: str
	save_path: str
	fps: int
	video_source: int
	logo_width: float
	countdown_duration: int


class State(Enum):
	LIVE = 1
	COUNTDOWN = 2
	RECORDING = 3
	PROCESSING = 4
	REPLAY = 5


logState = State.LIVE


logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler('debug.log'),  # Schreibt alles in diese Datei
		logging.StreamHandler(),  # Zeigt es trotzdem in der Konsole
	],
)

logging.info('App started')

# read config from yaml file


def read_config():
	logging.info('Read config from config.toml')
	with open('config.toml', 'rb') as f:
		config_dict = tomllib.load(f)
		try:
			config = Config(**config_dict)
		except Exception as e:
			logging.error(f'Error occurred while parsing config: {e}')
			raise e
	return config


def initialize_video(config: Config):
	logging.info('Initializing video capture')

	# initialize video capture

	# Nutze CAP_V4L2 explizit
	video = cv2.VideoCapture(config.video_source, cv2.CAP_V4L2)

	# MJPG ist oft die Ursache für die Korruption bei Billig-Webcams/alten Controllern
	# Wenn möglich, versuche YUYV (Standard weglassen) oder erhöhe den Buffer
	video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

	# video = cv2.VideoCapture(config.video_source)
	# video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
	video.set(cv2.CAP_PROP_FPS, config.fps)
	video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

	# we create a named window with the fullscreen property
	cv2.namedWindow('Lifter', cv2.WND_PROP_FULLSCREEN)
	# we get the actual FPS from the camera, which is important for accurate timing and recording,
	# cv2.setWindowProperty('Lifter', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  #
	# actual_fps = video.get(cv2.CAP_PROP_FPS) or 60.0
	# Buffer-Größe reduzieren, um Latenz/Korruption zu minimieren
	# video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	return video


def generate_qr(url):
	logging.info('Generating QR code for URL: %s', url)
	qr = qrcode.QRCode(box_size=6, border=2)
	qr.add_data(url)
	qr.make(fit=True)
	img: PilImage = qr.make_image(fill_color='black', back_color='white')  # type: ignore
	img: Image = img.convert('RGB')
	logging.info('QR code generated successfully')
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def add_logo(config: Config, frame: np.ndarray):

	frame_height, frame_width = frame.shape[:2]

	x = int(frame_width * 0.02)
	y = int(frame_height * 0.05)

	logo_raw = cv2.imread('static/NORDLICHT.png', cv2.IMREAD_UNCHANGED)
	if logo_raw is None:
		logging.error('Failed to load logo image')
		raise Exception('Failed to load logo image')

	# logo_raw.shape
	# (560, 1400, 4)
	orig_logo_height = logo_raw.shape[0]
	orig_logo_width = logo_raw.shape[1]

	scaled_logo_width = int(config.logo_width * frame_height)
	scaled_logo_height = int(orig_logo_height * scaled_logo_width / orig_logo_width)

	# 1. Skalieren (z.B. auf 200 Pixel Breite)
	# ratio = 200 / orig_logo_width
	# dim = (200, int(orig_logo_height * ratio))
	logo_resized = cv2.resize(
		logo_raw, (scaled_logo_width, scaled_logo_height), interpolation=cv2.INTER_AREA
	)

	# 2. Drehen (passend zu deinem Monitor gegen den Uhrzeigersinn)
	# Wenn der Monitor nach links gekippt ist, drehen wir das Logo nach rechts
	logo = cv2.rotate(logo_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)

	# 1. Prüfen, ob das Overlay einen Alpha-Kanal hat (4 Kanäle)
	if logo.shape[2] != 4:
		# Falls kein Alpha-Kanal da ist, normales Slicing (dein aktueller Stand)
		h, w = logo.shape[:2]
		frame[y : y + h, x : x + w] = logo
		return frame

	# 2. Alpha-Kanal extrahieren und normalisieren (0.0 bis 1.0)
	alpha_mask = logo[:, :, 3] / 255.0
	logo_color = logo[:, :, :3]

	# 3. Den Bereich im Hintergrund definieren, wo das Logo hin soll
	h, w = logo.shape[:2]

	# Sicherheitscheck für Bildgrenzen
	if y + h > frame.shape[0] or x + w > frame.shape[1]:
		return frame

	roi = frame[y : y + h, x : x + w]

	# 4. Das Blending berechnen:
	# Ergebnis = (Logo * Alpha) + (Hintergrund * (1 - Alpha))
	for c in range(0, 3):
		roi[:, :, c] = logo_color[:, :, c] * alpha_mask + roi[:, :, c] * (1.0 - alpha_mask)

	return frame


def draw_responsive_text(frame, text, rel_y=0.5, rel_x=0.5):
	h, w = frame.shape[:2]

	# Schriftgröße basierend auf der Bildhöhe (z.B. 2% der Höhe)
	font_scale = h * 0.0015
	thickness = max(1, int(h * 0.002))
	font = cv2.FONT_HERSHEY_SIMPLEX

	# Größe berechnen
	(text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

	# Position berechnen (rel_x=0.5 ist Mitte)
	x = int(w * rel_x - text_w / 2)
	y = int(h * rel_y + text_h / 2)

	# Schatten für bessere Lesbarkeit (optional)
	cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness)
	# Haupttext
	cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def get_relative_font_scale(frame, percent=0.05):
	"""
	Berechnet fontScale basierend auf der Frame-Breite.
	0.05 entspricht 5% der Breite.
	"""
	width = frame.shape[1]
	# Ein guter Basiswert: Bei 1920px Breite ist fontScale 1.0 oft ca. 2% der Breite.
	# Wir skalieren das nun linear:
	return (width / 1000) * (percent * 10)


def prepare_rotated_text(
	text, font, frame_width: int, percent_width: float, color, thickness, angle
):
	"""
	Berechnet die Schriftgröße basierend auf einem Prozentsatz der Frame-Breite.
	percent_of_width: z.B. 0.05 für 5% der Breite
	"""
	# 1. Berechne font_scale:
	# Ein fontScale von 1.0 bei 1000px Breite entspricht ca. 22-25px Höhe.
	# Wir skalieren das so, dass 'percent_of_width' die Zielgröße steuert.
	font_scale = (frame_width / 1000) * (percent_width * 20)

	# 2. Berechne Dicke proportional zur Breite (mindestens 1)
	final_thickness = max(thickness, 1, int(frame_width * 0.005))

	# Textgröße für das temporäre Bild berechnen
	text_size, _ = cv2.getTextSize(text, font, font_scale, final_thickness)
	tw, th = text_size

	# Quadratische Leinwand für die Rotation
	side = int((tw**2 + th**2) ** 0.5) + 20
	text_img = np.zeros((side, side, 3), dtype=np.uint8)

	# Text mittig zeichnen
	cv2.putText(
		text_img, text, ((side - tw) // 2, (side + th) // 2), font, font_scale, color, thickness
	)

	# Rotationsmatrix und Warp
	M = cv2.getRotationMatrix2D((side // 2, side // 2), 90, 1.0)
	rotated_img = cv2.warpAffine(text_img, M, (side, side))

	# Tight Crop (wie zuvor besprochen), um Offsets am Rand zu vermeiden
	gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
	coords = cv2.findNonZero(gray)
	if coords is not None:
		x_c, y_c, w_c, h_c = cv2.boundingRect(coords)
		tight_img = rotated_img[y_c : y_c + h_c, x_c : x_c + w_c]
	else:
		tight_img = rotated_img

	return tight_img


def apply_cached_text(frame, cached_img, relative_position):
	x = int(frame.shape[1] * relative_position[0])
	# compute y from bottom, not from top, because the text is rotated
	# y = int(frame.shape[0] * relative_position[1])
	y = int(frame.shape[0] * (1 - relative_position[1])) - cached_img.shape[0]
	h_rot, w_rot = cached_img.shape[:2]
	h_frame, w_frame = frame.shape[:2]

	# Out-of-bounds check
	y_end = min(y + h_rot, h_frame)
	x_end = min(x + w_rot, w_frame)
	h_real = y_end - y
	w_real = x_end - x

	if h_real > 0 and w_real > 0:
		roi = frame[y:y_end, x:x_end]
		text_part = cached_img[0:h_real, 0:w_real]

		# Effizientes Überlagern ohne Thresholding im Loop
		# Überall wo der Text-Part nicht schwarz (0) ist, ersetzen wir das Frame-Pixel
		mask = np.any(text_part > 0, axis=-1)
		roi[mask] = text_part[mask]


def prepare_rotated_qr(url, angle):
	logging.info(f'Generating and rotating QR code: {angle}°')

	# 1. QR generieren (deine Funktion)
	qr_bgr = generate_qr(url)

	# 2. Rotieren
	h, w = qr_bgr.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)

	# Da QR-Codes Quadrate sind, reicht w, h als Zielgröße
	rotated_qr = cv2.warpAffine(qr_bgr, M, (w, h), borderValue=(0, 0, 0))

	return rotated_qr


def apply_cached_qr(frame, cached_qr, relative_position):
	h_f, w_f = frame.shape[:2]
	h_q, w_q = cached_qr.shape[:2]

	# Position berechnen
	x = int(w_f * relative_position[0])
	y = int(h_f * (1 - relative_position[1])) - h_q

	# Clipping / Bounds Check
	y_end = min(y + h_q, h_f)
	x_end = min(x + w_q, w_f)

	# Tatsächliche Dimensionen im Frame (falls am Rand abgeschnitten)
	h_real = y_end - y
	w_real = x_end - x

	if h_real > 0 and w_real > 0:
		# Wir kopieren einfach das gesamte Rechteck drüber (ohne Maske!)
		frame[y:y_end, x:x_end] = cached_qr[0:h_real, 0:w_real]


def main_loop(video: cv2.VideoCapture, config: Config):
	logging.info('Starting main loop')

	# 1. Den allerersten Frame holen, um die Maße zu kennen
	success, init_frame = video.read()
	if not success:
		logging.error('Kamera liefert keine Daten für Initialisierung')
		return

		# Maße detektieren
	frame_height, frame_width = init_frame.shape[:2]
	logging.info(f'UI Skalierung basiert auf Frame-Breite: {frame_width}px')

	frame_grabbed: bool
	# frame:np.ndarray

	state = State.LIVE

	recording_text = prepare_rotated_text(
		text='RECORDING',
		font=cv2.FONT_HERSHEY_SIMPLEX,
		frame_width=frame_width,
		# scale=3.0,
		percent_width=0.1,  # 15% der Frame-Breite
		color=(0, 0, 255),
		thickness=6,
		angle=90,
	)

	# AUßERHALB des Loops:
	cached_qr = prepare_rotated_qr(config.server_url, 90)  # 90 Grad für Hochkant-Monitor

	countdown_timer = 0.0
	while True:
		if state == State.LIVE:
			frame_grabbed, frame = video.read()

			if not frame_grabbed:
				logging.error('Failed to grab frame from video source')
				raise RuntimeError('Failed to grab frame from video source')

			add_logo(config, frame)

			# draw_responsive_text(frame, 'Scan the QR code to join the game!', rel_y=0.9)
			apply_cached_text(
				frame,
				recording_text,
				relative_position=(0.02, 0.02),
			)

			apply_cached_qr(
				frame,
				cached_qr,
				relative_position=(0.88, 0.8),  # Mitte, mit etwas Abstand zum Rand
			)

			# show the frame in fullscreen
			cv2.imshow('Lifter', frame)

		elif state == State.COUNTDOWN:
			current_time = time.time()
			remaining = config.countdown_duration - int(current_time - countdown_timer)
			if remaining > 0:
				frame_grabbed, frame = video.read()
				if not frame_grabbed:
					logging.error('Failed to grab frame during countdown')
					raise RuntimeError('Failed to grab frame during countdown')

				add_logo(config, frame)
				# draw_responsive_text(frame, f'Starting in {remaining}...', rel_y=0.5)

				countdown_text = prepare_rotated_text(
					text=f'{remaining}',
					font=cv2.FONT_HERSHEY_SIMPLEX,
					# scale=12.0,
					frame_width=frame_width,
					percent_width=0.6,
					color=(0, 255, 255),
					thickness=30,
					angle=90,
				)

				# print(countdown_text.shape[0], frame.shape[0])
				apply_cached_text(
					frame,
					countdown_text,
					relative_position=(
						0.5 - (countdown_text.shape[1] / frame.shape[1]) / 2,  # X von links, Mitte
						0.5 - (countdown_text.shape[0] / frame.shape[0]) / 2,
					),  # Mitte
				)
				cv2.imshow('Lifter', frame)

			else:
				logging.info('Countdown finished, switching to RECORDING state')
				state = State.RECORDING

		elif state == State.RECORDING:
			# Hier würdest du den Aufnahme- und Upload-Logik implementieren
			# Für dieses Beispiel bleiben wir einfach im LIVE-Modus
			logging.info('Recording started (placeholder), switching back to LIVE state')
			state = State.LIVE

			# Optional: Hier könntest du auch eine kurze "Recording finished" Nachricht anzeigen

		key = cv2.waitKeyEx(1)  # & 0xFF
		# print(key)
		if key == ord('q'):
			break

		if state == State.LIVE and key == ord(' '):
			logging.info('Switching to RECORDING state')
			state = State.COUNTDOWN
			countdown_timer = time.time()

	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	config = read_config()

	video = initialize_video(config)

	main_loop(video, config)
