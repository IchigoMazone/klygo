"""Convert multiple raw source trees into one flat YOLO dataset."""

from tempfile import TemporaryDirectory
from PIL import Image

from klygo import datasets, files, media


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "raw"
    target = root / "flat"
    for batch, stem in (("batch_a", "apple_1"), ("batch_b", "banana_1")):
        images = source / batch / "images"
        labels = source / batch / "labels"
        files.mkdir(images)
        files.mkdir(labels)
        media.save(images / f"{stem}.jpg", Image.new("RGB", (64, 48), "white"), verbose=False)
        files.save(labels / f"{stem}.txt", ["0 0.5 0.5 0.4 0.4"], verbose=False)

    output = datasets.convert(
        source,
        target,
        tree={"pictures": "batch_*/images", "annotations": "batch_*/labels"},
        classes={0: "fruit"},
        on_conflict="rename",
    )
    print(datasets.info(output, verbose=False))
