"""Demonstrate media.MediaFrames."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    frames = media.load(root)
    print(isinstance(frames, media.MediaFrames), frames.total_frames, frames.loaded_count)
