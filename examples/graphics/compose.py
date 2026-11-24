"""Runnable example for graphics.compose."""

from PIL import Image
from klygo import graphics

items = graphics.compose([graphics.cell(Image.new("RGB", (64, 48), "white"), width=100)])
print(len(items), items[0].source.size)
