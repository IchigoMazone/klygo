"""Create deterministic train, validation, and test partitions."""

from tempfile import TemporaryDirectory

from klygo import datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(root / "source")
    output = datasets.partition(source, root / "partitioned", (0.5, 0.25, 0.25), verbose=False)
    assert files.is_dir(output / "images" / "train")
    assert datasets.info(output)["total_images"] == 4
