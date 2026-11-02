"""Flatten and load every image and annotation from a dataset archive."""

from tempfile import TemporaryDirectory

from klygo import archive, datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(files.join(root, "source"))
    packed = files.join(root, "dataset.zip")
    archive.compress(source, packed, verbose=False)

    flat = datasets.unpartition(packed, files.join(root, "flat"), verbose=False)
    results = datasets.load(flat, format="yolo", verbose=False)
    print("Images:", len(results))
    print("Objects:", results.total_objects)
    print("First image:", results[0].metadata["relative_path"])
    print("Boxes in pixels:", results[0].boxes)

# Annotation dimensions remain available without decoding source pixels.
print("First image size:", results[0].image_size)
