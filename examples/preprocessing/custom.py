"""Demonstrate preprocessing.custom."""

import os
import sys

# Keep inspect.py from shadowing Python's inspect module when run directly.
if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(os.path.dirname(__file__)):
    sys.path.pop(0)

from PIL import Image
from klygo import preprocessing as p, media

image = Image.new("RGB", (64, 48), (120, 80, 40))
operation = p.custom(lambda value: value.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
print(operation(image).size)

resize = p.custom(
    lambda value: value.resize((32, 24)),
    name="half_size",
    changes_geometry=True,
    inverse=lambda point, step: (point[0] * 2, point[1] * 2),
)
print(resize(image).size)
