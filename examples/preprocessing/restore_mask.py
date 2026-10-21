"""Demonstrate preprocessing.restore_mask."""

import os
import sys

# Keep inspect.py from shadowing Python's inspect module when run directly.
if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(os.path.dirname(__file__)):
    sys.path.pop(0)

from PIL import Image
from klygo import preprocessing as p, media

image = Image.new("RGB", (64, 48), (120, 80, 40))
import tempfile
from klygo import files
with tempfile.TemporaryDirectory() as temporary:
    source = media.save(files.path(temporary) / "input.png", image)
    lazy = media.load(source)[0]
    processed = p.resize((32, 24))(lazy)
    p.materialize(processed, output="pil")
    history = p.get_trace(processed)
    print(p.restore_mask(Image.new("L", (32, 24), 255), history).size)
