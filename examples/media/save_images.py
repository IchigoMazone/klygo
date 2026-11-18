"""Demonstrate media.save_images."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (80, 40), "red"), Image.new("RGB", (40, 80), "blue")]
    media.save_images(root / "frames", frames, extension=".png", size=(64, 64))
    paths = files.find(root / "frames", recursive=False)
    print([files.name(path) for path in paths])
    print([media.info(path)["size"] for path in paths])
