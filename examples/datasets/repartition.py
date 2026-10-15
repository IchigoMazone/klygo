"""Replace existing dataset split ratios."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import archive, datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_dataset(root / "source")
    output = datasets.repartition(source, root / "repartitioned.zip", (0.5, 0.25), verbose=False)
    assert archive.is_archive(output)
    assert datasets.get_dataset_info(output)["total_images"] == 4
