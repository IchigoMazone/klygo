"""Demonstrate media.probe."""

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
    print(media.probe(source)["type"], media.probe(source)["size"])
