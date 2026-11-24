"""Runnable example for graphics.grid."""

from PIL import Image
from klygo import graphics

items = graphics.compose([graphics.cell(Image.new("RGB", (64, 48), "red")), graphics.cell(Image.new("RGB", (64, 48), "blue"))])
graphics.grid(items, cols=2, show=False)
