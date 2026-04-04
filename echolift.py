import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import qrcode
from cv2.typing import MatLike
from PIL.Image import Image
from qrcode.image.pil import PilImage

SERVER_URL = os.getenv('LIFTER_SERVER_URL', 'http://192.168.178.82:8000')
SAVE_PATH = os.getenv('LIFTER_SAVE_PATH', 'static/videos')


def generate_qr(url):
	print('generating QR code for URL:', url)
	qr = qrcode.QRCode(box_size=4, border=2)
	qr.add_data(url)
	qr.make(fit=True)
	img: PilImage = qr.make_image(fill_color='black', back_color='white')  # type: ignore
	img: Image = img.convert('RGB')
	print("all ok")
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# QR-Code generieren (wir machen ihn ca. 150x150 Pixel groß)
# qr_img = generate_qr(SERVER_URL)
# qr_h, qr_w, _ = qr_img.shape

os.makedirs(SAVE_PATH, exist_ok=True)


# EINSTELLUNGEN
VIDEO_SOURCE = 1
DELAY_SECONDS = 10
FPS = 30
WIDTH, HEIGHT = 1920, 1080


# initialize video capture
video = cv2.VideoCapture(VIDEO_SOURCE)

# MJPG is a good choice for webcam capture, as it provides a good balance between quality
# and performance
# usb logitech
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

# we set the desired resolution for the capture
video.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# we set the desired FPS for the capture, which is important for ensuring smooth video
# and good temporal resolution for analysis. The C930e can handle 1080p at 30
video.set(cv2.CAP_PROP_FPS, FPS)
window_name = 'Lifter'
# we create a named window with the fullscreen property
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# we get the actual FPS from the camera, which is important for accurate timing and recording,
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  #
actual_fps = video.get(cv2.CAP_PROP_FPS) or 60.0


# we initialize the video writer variable, which will be used later for recording the video.
video_writer: cv2.VideoWriter | None = None

# we initialize the replay video capture variable, which will be used later for
# playing back the recorded video in a loop.
replay_video: cv2.VideoCapture | None = None


# FPS-calculation setup
prev_frame_time = 0
new_frame_time = 0
frame_count = 0

# various variables for managing the recording and replay process,
# such as filenames, timers, and frame counts.
temp_filename = ''
replay_filename = ''
countdown_timer = 0
start_time = 0
ffmpeg_process = None

state = 'LIVE'  # possible states: LIVE, COUNTDOWN, RECORDING, REPLAY

display_frame = None
frame = None


# WICHTIG: -1 lädt das PNG inklusive Transparenz (BGRA)
"""
logo = cv2.imread('static/NORDLICHT.png', -1)

# Skalieren (falls nötig) unter Beibehaltung der 4 Kanäle
logo = cv2.resize(logo, (300, 300))

# Drehen (gegen den Uhrzeigersinn für dein Setup)
logo = cv2.rotate(logo, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Logo um 90 Grad GEGEN den Uhrzeigersinn drehen
# (Passend zur Drehung deines Monitors)
rotated_logo = cv2.rotate(logo, cv2.ROTATE_90_COUNTERCLOCKWISE)
"""

# -1 ist entscheidend für den Alpha-Kanal (Transparenz)
logo_raw = cv2.imread('static/NORDLICHT.png', cv2.IMREAD_UNCHANGED)

# 1. Skalieren (z.B. auf 200 Pixel Breite)
ratio = 200 / logo_raw.shape[1]
dim = (200, int(logo_raw.shape[0] * ratio))
logo_resized = cv2.resize(logo_raw, dim, interpolation=cv2.INTER_AREA)

# 2. Drehen (passend zu deinem Monitor gegen den Uhrzeigersinn)
# Wenn der Monitor nach links gekippt ist, drehen wir das Logo nach rechts
logo_final = cv2.rotate(logo_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)


def overlay_transparent(background, overlay, x, y):
	# 1. Prüfen, ob das Overlay einen Alpha-Kanal hat (4 Kanäle)
	if overlay.shape[2] != 4:
		# Falls kein Alpha-Kanal da ist, normales Slicing (dein aktueller Stand)
		h, w = overlay.shape[:2]
		background[y : y + h, x : x + w] = overlay
		return background

	# 2. Alpha-Kanal extrahieren und normalisieren (0.0 bis 1.0)
	alpha_mask = overlay[:, :, 3] / 255.0
	overlay_color = overlay[:, :, :3]

	# 3. Den Bereich im Hintergrund definieren, wo das Logo hin soll
	h, w = overlay.shape[:2]

	# Sicherheitscheck für Bildgrenzen
	if y + h > background.shape[0] or x + w > background.shape[1]:
		return background

	roi = background[y : y + h, x : x + w]

	# 4. Das Blending berechnen:
	# Ergebnis = (Logo * Alpha) + (Hintergrund * (1 - Alpha))
	for c in range(0, 3):
		roi[:, :, c] = overlay_color[:, :, c] * alpha_mask + roi[:, :, c] * (1.0 - alpha_mask)

	return background


def apply_gym_filter(frame):
	# soft gaussian blur with a small kernel, just to smooth out the noise
	# without losing too much detail
	frame = cv2.GaussianBlur(frame, (3, 3), 0)
	# convert to LAB color space for better contrast manipulation
	lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
	# split the LAB channels to apply CLAHE only on the L channel (lightness),
	# so that we enhance contrast without affecting colors
	l, a, b = cv2.split(lab)  # noqa: E741
	# smaller gridsize for more local contrast enhancement,
	# but not too small to avoid over-sharpening
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
	# apply CLAHE to the L channel to enhance contrast, which can help in making the lifter
	# more visible against the background, especially in varying lighting conditions
	cl = clahe.apply(l)
	# merge the enhanced L channel back with the original A and B channels to keep the
	# color information intact while improving the overall contrast of the image
	limg = cv2.merge((cl, a, b))
	# convert back to BGR color space for further processing and display
	frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	# we use a larger sigma for a stronger blur effect, which will be used to create
	# a glow around the lifter when we blend it back with the original frame
	gaussian_3 = cv2.GaussianBlur(frame, (0, 0), 2.0)

	# blend the original frame with the blurred version to create a glow effect around
	# the lifter, which can help in making them stand out more against the background,
	# especially in dynamic gym environments where there may be a lot of visual
	# noise and distractions
	frame = cv2.addWeighted(frame, 1.5, gaussian_3, -0.5, 0)

	return frame


def put_text(
	frame: MatLike,
	text: str,
	position_percent: tuple[float, float],
	font_scale: float = 1,
	color=(255, 255, 255),
	thickness=2,
	font_face=cv2.FONT_HERSHEY_SIMPLEX,
	line_type=cv2.LINE_AA,
):
	h, w = frame.shape[:2]
	position = (int(w * position_percent[0]), int(h * position_percent[1]))
	cv2.putText(
		frame,
		text,
		position,
		font_face,
		font_scale,
		color,
		thickness,
		line_type,
	)


def draw_rotated_text(
	frame, text, position_percent: tuple[float, float], font, scale, color, thickness, angle
):
	# 1. Textgröße berechnen
	text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
	tw, th = text_size

	h, w = frame.shape[:2]
	position = (int(w * position_percent[0]), int(h * position_percent[1]))

	# Sicherheitscheck: Falls der Text leer ist
	if tw <= 0 or th <= 0:
		return frame

	# 2. Ein ausreichend großes temporäres Bild erstellen (Quadratisch, um Drehung zu erlauben)
	# Wir nehmen die Diagonale als Seitenlänge, damit der Text beim Drehen nicht abgeschnitten wird
	side = int((tw**2 + th**2) ** 0.5) + 20
	text_img = np.zeros((side, side, 3), dtype=np.uint8)

	# Text in die Mitte des schwarzen Bildes schreiben
	text_x = (side - tw) // 2
	text_y = (side + th) // 2
	cv2.putText(text_img, text, (text_x, text_y), font, scale, (255, 255, 255), thickness)

	# 3. Das Bild drehen
	center = (side // 2, side // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated_text = cv2.warpAffine(text_img, M, (side, side))

	# FEHLER-CHECK: Sicherstellen, dass rotated_text existiert
	if rotated_text is None:
		return frame

	# 4. Maske erstellen (wo ist Text?)
	gray = cv2.cvtColor(rotated_text, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

	# 5. Region of Interest (ROI) im Hauptbild finden
	y_start, x_start = position

	h_rot, w_rot = rotated_text.shape[:2]

	# Prüfen, ob die Koordinaten im Bild liegen (Crop gegen Out-of-Bounds)
	y_end = min(y_start + h_rot, h)
	x_end = min(x_start + w_rot, w)

	# Tatsächliche Größe der Fläche, die wir beschreiben können
	h_real = y_end - y_start
	w_real = x_end - x_start

	if h_real > 0 and w_real > 0:
		print(f'Placing text at: ({x_start}, {y_start}), size: ({w_real}x{h_real})')
		# Nur den Teil des Textes nehmen, der auch ins Bild passt
		roi = frame[y_start:y_end, x_start:x_end]
		mask_part = mask[0:h_real, 0:w_real]

		# Farbe auf die Masken-Pixel anwenden
		roi[mask_part > 0] = color

	return frame


print("Station bereit.  für Start/Stopp, 'q' zum Beenden.")


qr_img = None

while True:
	# 1. Frame von Kamera lesen (nur wenn wir nicht im Replay sind)
	qr_img = None  # QR-Code zurücksetzen, damit er im nächsten Replay neu generiert wird

	if state != 'REPLAY' and state != 'PROCESSING':
		ret, frame = video.read()

		# draw_rotated_text(
		# frame, 'BEREIT', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, 90
		# )

		# Gedrehtes Logo einblenden (z.B. oben rechts)
		try:
			# Koordinaten berechnen (Beispiel: Oben Rechts)
			# x = Gesamtbreite - Logobreite - Abstand
			# y = Abstand von oben
			h_logo, w_logo = logo_final.shape[:2]
			# x_pos = frame.shape[1] - w_logo - 30
			x_pos = 30
			y_pos = 30

			# Die Funktion einfügen
			frame = overlay_transparent(frame, logo_final, x_pos, y_pos)
		except Exception as e:
			print(f'Fehler beim Einblenden des Logos: {e}')

		# Wenn Daten korrupt sind, liefert OpenCV oft ein leeres Bild oder None
		if not ret or frame is None or frame.size == 0:
			continue  # Überspringe diesen Frame einfach und nimm den nächsten

		frame = apply_gym_filter(frame)
		if not ret:
			print("break2")
			break
		display_frame = frame.copy()

		# FPS calculation
		new_frame_time = time.time()
		# fps = 1 / (time difference between current frame and previous frame)

		fps = 1 / (new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		# show FPS in top-left corner

		"""
		put_text(
			display_frame,
			f'FPS: {int(fps)}',
			(0.05, 0.05),
			font_scale=1,
			color=(0, 255, 0),
			thickness=2,
		)
		"""

	current_time = time.time()

	# countdown
	if state == 'COUNTDOWN':
		remaining = 2 - int(current_time - countdown_timer)
		if remaining > 0:
			font = cv2.FONT_HERSHEY_DUPLEX
			text = str(remaining)
			# Dicke (thickness) auf 15 für massiven Look

			if display_frame is not None and display_frame.any():
				"""
				put_text(
					display_frame,
					text,
					(0.3, 0.5),
					font_scale=7,
					color=(0, 255, 255),
					thickness=15,
				)
				"""

				try:
					draw_rotated_text(
						display_frame,
						text,
						(0.01, 0.3),
						cv2.FONT_HERSHEY_SIMPLEX,
						35,
						(0, 255, 255),
						35,
						90,
					)
				except Exception as e:
					print(f'Fehler beim Zeichnen des Textes: {e}')
		else:
			# Wechsel zu RECORDING
			state = 'RECORDING'
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			temp_filename = os.path.join(SAVE_PATH, f'lift_{timestamp}_temp.avi')
			replay_filename = os.path.join(SAVE_PATH, f'lift_{timestamp}.mp4')
			#fourcc = cv2.VideoWriter.fourcc(*'MJPG')

			h, w = frame.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			video_writer = cv2.VideoWriter(
				temp_filename,  # the temporary raw video file (MJPG)
				fourcc,  # this is the codec for MJPG
				30.0,  # the actual FPS
				(w,h),
			)
			start_time = time.time()
			frame_count = 0

	# --- LOGIK: RECORDING ---
	elif state == 'RECORDING':
		if frame is not None and frame.any() and video_writer:
			video_writer.write(frame)
		frame_count += 1
		if display_frame is not None and display_frame.any():
			"""
			put_text(
				display_frame,
				'REC',
				(0.05, 0.1),
				font_scale=1,
				color=(0, 0, 255),
				thickness=2,
			)
			"""

			try:
				draw_rotated_text(
					display_frame,
					'REC',
					(0.2, 0.00001),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(0, 0, 255),
					10,
					90,
				)
			except Exception as e:
				print(f'Fehler beim Zeichnen des Textes: {e}')

	# --- LOGIK: REPLAY ---
	elif state == 'REPLAY':
		if not replay_video:
			continue
		ret, replay_frame = replay_video.read()

		if not ret:
			# end of video, reset to beginning
			replay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue

		# display QR code
		margin = 30
		# place QR code in the top-right corner with a margin
		if qr_img is None:
			qr_img = generate_qr(SERVER_URL + '?video=' + os.path.basename(replay_filename))
			qr_h, qr_w, _ = qr_img.shape

			replay_frame[margin : margin + qr_h, WIDTH - qr_w - margin : WIDTH - margin] = qr_img

		# place "LOOP REPLAY"

		try:
			draw_rotated_text(
				replay_frame,
				'REPLAY - SPACE TO STOP',
				(0.2, 1.45),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.25,
				(0, 165, 255),
				3,
				90,
			)
		except Exception as e:
			print(f'Fehler beim Zeichnen des Textes: {e}')

		# this function is called only in REPLAY state, so we show the replay frame with QR code
		# and texts.
		# In LIVE and COUNTDOWN states, we show the normal display_frame without QR code.
		cv2.imshow(window_name, replay_frame)

		# we make a short delay here to allow the replay video to play at a reasonable
		# speed, and also to check for key presses to exit the replay.
		# BERECHNUNG DER WARTEZEIT:
		# Wenn das Original 30 FPS hatte und FFmpeg es verdoppelt hat (setpts=2.0),
		# müssen wir jetzt mit ca. 15 FPS abspielen.
		# Formel: 1000ms / (actual_fps / 2)
		# Bei 30 FPS Kamera: 1000 / 15 = 66ms

		wait_time = int(1000 / (actual_fps / 2))
		key = cv2.waitKey(wait_time) & 0xFF

		if key == ord(' '):
			replay_video.release()
			state = 'LIVE'
		elif key == ord('q'):
			break
		continue

	elif state == 'PROCESSING':
		time.sleep(0.5)  # Gib der CPU Luft zum Atmen

		ret, frame = video.read()  # Weiterhin Live-Bild lesen
		if ret:
			display_frame = apply_gym_filter(frame)
			# Overlay für den Lade-Status

			"""
			put_text(
				display_frame,
				'PROCESSING LIFT...',
				(0.3, 0.5),
				font_scale=1.2,
				color=(0, 165, 255),
				thickness=3,
			)
			"""

			try:
				draw_rotated_text(
					display_frame,
					'Processing Video...',
					(0.1, 0.5),
					cv2.FONT_HERSHEY_SIMPLEX,
					2,
					(0, 165, 255),
					4,
					90,
				)
			except Exception as e:
				print(f'Fehler beim Zeichnen des Textes: {e}')

			cv2.imshow(window_name, display_frame)

		# Prüfen, ob FFmpeg fertig ist
		# poll() ist None, solange der Prozess läuft
		 
		if ffmpeg_process:# and ffmpeg_process.poll() is not None:
			ffmpeg_result = ffmpeg_process.poll()
			print(f"ffmpeg result: {ffmpeg_result}")
			if ffmpeg_result is not None:
				if os.path.exists(temp_filename):
					os.remove(temp_filename)

			replay_video = cv2.VideoCapture(replay_filename)
			state = 'REPLAY'

		# Auch hier auf ' ' prüfen, falls man abbrechen will
		if cv2.waitKey(1) & 0xFF == ord(' '):
			print("break1")
			break
		continue

	# display the current frame (with overlays) in LIVE and COUNTDOWN states
	if display_frame is not None and display_frame.any():
		cv2.imshow(window_name, display_frame)

	# key press handling for LIVE and COUNTDOWN states (REPLAY state is handled separately above)
	key = cv2.waitKey(1) & 0xFF

	# handle " " key for starting/stopping recording, and 'q' for quitting the application

	#print(state)
	if key == ord(' '):
		if state == 'LIVE':
			state = 'COUNTDOWN'
			countdown_timer = time.time()
		elif state == 'RECORDING':
			# Aufnahme stoppen
			state = 'LIVE'  # Kurz auf Live, während wir konvertieren
			if video_writer:
				video_writer.release()
			duration = time.time() - start_time
			measured_fps = frame_count / duration

			# Konvertierung (synchron, da kurz)
			# Konvertierung mit eingebauter Zeitlupe (0.5x)
			cmd = [
				'ffmpeg',
				'-y',
				'-r',
				str(measured_fps),  # Input FPS
				'-i',
				temp_filename,  # Quelle 0: Video
				'-i',
				'static/NORDLICHT.png',  # Quelle 1: Logo
				'-filter_complex',
				# Schritt 1: Logo skalieren -> [logo]
				# Schritt 2: Video (0:v) verlangsamen -> [slow]
				# Schritt 3: [logo] über [slow] legen
				'[1:v]scale=200:-1[logo];[0:v]setpts=2.0*PTS[slow];[slow][logo]overlay=20:main_h-overlay_h-20',
				'-c:v',
				'libx264',
				'-preset',
				'ultrafast',  # WICHTIG: Damit du im Gym nicht ewig warten musst
				'-pix_fmt',
				'yuv420p',
				replay_filename,
			]

			cmd = [
				'ffmpeg',
				'-y',
				#'-r',
				#str(measured_fps or 30),  # Input FPS mit Fallback
				'-i',
				temp_filename,  # Einzige Quelle: Das Video inkl. Logo
				'-vf',
				'setpts=2.0*PTS',  # Nur noch die Zeitlupe (0.5x Speed)
				'-c:v',
				'libx264',
				'-preset',
				'ultrafast',  # Maximale Geschwindigkeit
				'-pix_fmt',
				'yuv420p',  # Standard-Format für maximale Kompatibilität
				replay_filename,
			]

			# Wir fügen 'nice -n 15' vor den eigentlichen Befehl
			cmd = [
				'nice', '-n', '15', 
				'ffmpeg', '-y',
				'-i', temp_filename,
				'-vf', 'setpts=2.0*PTS',
				'-c:v', 'libx264',
				'-preset', 'ultrafast',
				'-threads', '1',  # WICHTIG: Begrenze auf 1 Kern, damit 1-3 Kerne für Python frei bleiben
				'-pix_fmt', 'yuv420p',
				replay_filename,
			]

			"""
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # delete the temporary raw video file, we don't need it anymore
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            # start replay
            replay_video = cv2.VideoCapture(replay_filename)
            state = "REPLAY"
            """
			print('converting video with FFmpeg, please wait...')
			# Popen startet den Prozess, blockiert aber NICHT das Skript
			ffmpeg_process = subprocess.Popen(
				cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
			)

			print (f"ffmpeg strarted with code {ffmpeg_process.poll()}")

			

			# Sofort in den Replay-Modus springen geht jetzt nicht direkt,
			# da die Datei erst fertig sein muss.
			state = 'PROCESSING'  # Neuer Zwischenstatus

	# handle 'q' key to quit the application
	elif key == ord('q'):
		break


video.release()
cv2.destroyAllWindows()
