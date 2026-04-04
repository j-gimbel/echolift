import os
import subprocess
import time
from datetime import datetime

# read yaml config  files
import yaml

import cv2
import numpy as np
import qrcode
from cv2.typing import MatLike
from PIL.Image import Image
from qrcode.image.pil import PilImage


from pydantic import BaseModel

class Config(BaseModel):
	server_url: str
	save_path: str

# read config from yaml file
with open('config.yaml', 'r') as f:
	config_dict = yaml.safe_load(f)
	config = Config(**config_dict)

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
