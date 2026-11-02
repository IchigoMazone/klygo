"""Demonstrate preprocessing.from_input_spec."""

import os
import sys

# Keep inspect.py from shadowing Python's inspect module when run directly.
if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(os.path.dirname(__file__)):
    sys.path.pop(0)

from PIL import Image
from klygo import preprocessing as p, media

image = Image.new("RGB", (64, 48), (120, 80, 40))
pipeline = p.from_input_spec({})
print(type(pipeline).__name__, len(pipeline))
