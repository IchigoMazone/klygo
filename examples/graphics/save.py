"""Runnable example for graphics.save."""

from tempfile import TemporaryDirectory
from klygo import files, graphics

fig = graphics.partitions([{"split": "train", "images": 10, "ratio_pct": 100.0}], show=False, return_figure=True)
with TemporaryDirectory() as tmp:
    out = graphics.save(fig, files.path(tmp) / "chart.png")
    print(files.is_file(out))
