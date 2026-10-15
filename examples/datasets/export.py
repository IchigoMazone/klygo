"""Export normalized model results as a YOLO dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from klygo import datasets, files
from _support import ExampleDetector


with TemporaryDirectory() as directory:
    output = datasets.export(
        ExampleDetector(),
        Path(directory) / "generated",
        "yolo",
        [Image.new("RGB", (32, 24), "white")],
        classes=["object"],
        verbose=False,
    )
    assert files.is_file(output / "images" / "img_00001.jpg")
    assert files.is_file(output / "labels" / "img_00001.txt")
