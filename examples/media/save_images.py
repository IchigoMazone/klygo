"""Demonstrate media.save_images."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
    paths = media.save_images(root / "frames", frames, extension=".png")
    print([files.name(path) for path in paths])
