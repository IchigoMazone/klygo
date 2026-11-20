"""Runnable example for graphics.cell."""

from PIL import Image
from klygo import graphics

c = graphics.cell(Image.new("RGB", (64, 48), "white"), width=120, align="top-left", offset=(4, 4), title="Sample")
print(c.width, c.align)
