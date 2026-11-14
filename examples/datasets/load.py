"""Flatten and load every image and annotation from a dataset archive."""

from tempfile import TemporaryDirectory

from klygo import archive, datasets, files, metrics
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
    files.copy(flat / "labels", flat / "pseudo_labels", overwrite=True)
    sets = datasets.load(flat, labels_dir=["pseudo_labels", "labels"], verbose=False)
    sets.astype(32).pack().unpack()
    print("Label sets:", list(sets.keys()))  # pred, truth
    print(metrics.evaluate(sets["pred"], sets["truth"]))

# Annotation dimensions remain available without decoding source pixels.
print("First image size:", results[0].image_size)
