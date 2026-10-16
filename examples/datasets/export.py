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

    marker = output / "keep.txt"
    marker.write_text("previous output", encoding="utf-8")
    try:
        datasets.export(
            ExampleDetector(), output, "unknown", [Image.new("RGB", (32, 24))],
            classes=["object"], overwrite=True, verbose=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown formats must be rejected")
    assert marker.read_text(encoding="utf-8") == "previous output"
