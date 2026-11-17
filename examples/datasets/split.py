"""Create class-based and ratio-based dataset directories and ZIP archives."""

from tempfile import TemporaryDirectory

from PIL import Image

from klygo import archive, datasets, files, media
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(root / "source")

    for split_name in ("train", "val"):
        media.save(source / "images" / split_name / "duplicate.jpg", Image.new("RGB", (8, 8)), verbose=False)
        files.save(source / "labels" / split_name / "duplicate.txt", ["0 0.5 0.5 1 1"], verbose=False)
    datasets.split(source, root / "parts", ratios=(0.5, 0.5), verbose=False)
    outputs = files.list_entries(root / "parts")
    assert len(outputs) == 2
    assert all(files.is_dir(path) for path in outputs)
    assert sum(datasets.info(path)["total_images"] for path in outputs) == 6

    datasets.split(source, root / "archives" / "batch_*.zip", ratios=(0.5, 0.5), verbose=False)
    zipped = files.find(root / "archives", "*.zip")
    assert len(zipped) == 2
    assert all(archive.is_archive(path) for path in zipped)
