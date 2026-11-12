"""Demonstrate media.to_tensor."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    try:
        tensor = media.to_tensor(Image.new("RGB", (32, 24), "red"), normalize=True)
    except ImportError as exc:
        print(f"Optional extra required: {exc}")
    else:
        print(tuple(tensor.shape), str(tensor.dtype))
