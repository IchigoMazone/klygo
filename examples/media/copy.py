"""Demonstrate media.copy."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = root / "source.png"
    media.save(source, image)
    duplicate = root / "copy.png"
    media.copy(source, duplicate)
    print(media.info(duplicate)["size"])
