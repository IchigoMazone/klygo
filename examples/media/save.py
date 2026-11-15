"""Demonstrate media.save."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    destination = root / "image.png"
    media.save(destination, Image.new("RGB", (32, 24), "red"))
    print(media.info(destination)["size"])
