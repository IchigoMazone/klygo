"""Demonstrate media.to_pil."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    import numpy as np
    image = media.to_pil(np.zeros((24, 32, 3), dtype=np.uint8))
    print(image.size, image.mode)
