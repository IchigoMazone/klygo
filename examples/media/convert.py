"""Demonstrate media.convert."""

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
    target = root / "converted.jpg"
    media.convert(source, target)
    print(media.info(target)["format"])
