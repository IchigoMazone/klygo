"""Plot class counts from a temporary YOLO dataset."""

import tempfile

import matplotlib
from PIL import Image

from klygo import files, visual


matplotlib.use("Agg", force=True)
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    files.mkdir(root / "images")
    files.mkdir(root / "labels")
    files.save(root / "data.yaml", "names: [cat, dog]\nnc: 2\n")
    Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
    files.save(root / "labels" / "sample.txt", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    visual.plot_dataset_stats(root)
    print("Plotted two class counts")
