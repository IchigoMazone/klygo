"""Demonstrate media.save_video."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (80, 40), "red"), Image.new("RGB", (40, 80), "blue")]
    video = root / "clip.mp4"
    media.save_video(video, frames, fps=5, size=(64, 64))
    print(media.info(video)["size"], media.info(video)["frame_count"])
