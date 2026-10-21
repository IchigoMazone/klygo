"""Test fixture support for datasets test suite."""

from pathlib import Path
import tempfile
from PIL import Image

from klygo import archive, config, files, media


def ensure_test_data_zip(target_path: str = "test/data.zip") -> Path:
    """Ensure a valid YOLO zip dataset exists for testing, creating one if absent."""
    target = files.path(target_path)
    if files.exists(target):
        return Path(target)

    files.mkdir(files.parent(target), parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = files.path(tmp)
        raw_images = files.join(tmp_path, "images", "raw")
        raw_labels = files.join(tmp_path, "labels", "raw")
        files.mkdir(raw_images, parents=True, exist_ok=True)
        files.mkdir(raw_labels, parents=True, exist_ok=True)

        for i in range(10):
            img_p = files.join(raw_images, f"frame_{i}.jpg")
            media.save(img_p, Image.new("RGB", (32, 24), "white"), verbose=False)
            lbl_p = files.join(raw_labels, f"frame_{i}.txt")
            files.save(lbl_p, ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)

        config.save(
            files.join(tmp_path, "data.yaml"),
            {"names": ["apple"], "nc": 1, "path": str(tmp_path)},
            overwrite=True,
            verbose=False,
        )
        archive.compress(tmp_path, target, verbose=False)
    return Path(target)
