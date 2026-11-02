"""Demonstrate media.VideoReader."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    video = media.save_video(root / "clip.mp4", [Image.new("RGB", (32, 24), "red")], fps=5)
    reader = media.VideoReader(video)
    try:
        print(reader.read_frame(0).shape)
    finally:
        reader.close()
