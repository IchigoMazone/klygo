"""Demonstrate media.to_array."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    array = media.to_array(Image.new("RGB", (32, 24), "red"))
    print(array.shape, array.dtype)
