"""Demonstrate preprocessing.colorize."""

import os
import sys

# Keep inspect.py from shadowing Python's inspect module when run directly.
if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(os.path.dirname(__file__)):
    sys.path.pop(0)

from PIL import Image
from klygo import preprocessing as p, media

image = Image.new("RGB", (64, 48), (120, 80, 40))
operation = p.colorize("black", "white")
result = operation(image)
print(operation.name, type(result).__name__)
