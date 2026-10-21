"""Demonstrate postprocessing.drop."""

import sys

# Keep select.py from shadowing Python's standard select module.
sys.path.pop(0)

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(image, [Box(0, "cat", 0.9, [-5, 0, 50, 50]), Box(0, "cat", 0.8, [0, 0, 48, 48]), Box(1, "dog", 0.4, [60, 10, 90, 60])])
print(post.drop(frame, labels=["dog"]).labels)