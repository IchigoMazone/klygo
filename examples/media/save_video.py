"""Demonstrate media.save_video."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
    video = media.save_video(root / "clip.mp4", frames, fps=5)
    print(media.probe(video)["frame_count"])
