"""Rename classes and move class IDs."""

from tempfile import TemporaryDirectory

from klygo import config, datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(root / "source")
    output = datasets.remap_classes(
        source,
        root / "remapped",
        {"apple": "fruit", 1: 3},
        verbose=False,
    )

    metadata = config.load(output / "data.yaml", verbose=False)
    assert metadata.names == {0: "fruit", 3: "banana"}
    assert datasets.info(output)["nc"] == 4
    try:
        datasets.remap_classes(source, root / "collision", {1: 0}, verbose=False)
    except ValueError as error:
        assert "multiple class names" in str(error)
    else:
        raise AssertionError("Conflicting class IDs must be rejected")
