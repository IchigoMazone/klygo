import unittest
import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image

from klygo.datasets import get_dataset_info, info
from klygo.archive import compress, extract
from klygo import config, files, media


def _create_sample_dataset(root: Path) -> Path:
    img_dir = root / "images" / "raw"
    lbl_dir = root / "labels" / "raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        media.save(img_dir / f"frame_{i}.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(lbl_dir / f"frame_{i}.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        root / "data.yaml",
        {"names": ["apple"], "nc": 1, "path": str(root)},
        overwrite=True,
        verbose=False,
    )
    return root


class TestInfo(unittest.TestCase):
    def test_alias(self):
        self.assertIs(get_dataset_info, info)

    def test_info_valid_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "source_ds"
            _create_sample_dataset(ds_dir)
            zip_path = Path(tmpdir) / "sample.zip"
            compress(ds_dir, zip_path, verbose=False)
            res = get_dataset_info(zip_path, verbose=False)
            self.assertEqual(res["nc"], 1)
            self.assertEqual(res["classes"], ["apple"])
            self.assertEqual(res["total_images"], 4)
            self.assertEqual(res["total_labels"], 4)
            self.assertEqual(res["unlabeled_images"], 0)
            self.assertIn("raw", res["splits"])
            self.assertEqual(res["splits"]["raw"], 4)

    def test_info_valid_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "source_ds"
            _create_sample_dataset(ds_dir)
            res = get_dataset_info(ds_dir, verbose=False)
            self.assertEqual(res["nc"], 1)
            self.assertEqual(res["classes"], ["apple"])
            self.assertEqual(res["total_images"], 4)
            self.assertEqual(res["total_labels"], 4)
            self.assertEqual(res["unlabeled_images"], 0)
            self.assertIn("raw", res["splits"])

    def test_info_nonexistent(self):
        print("Testing get_dataset_info on a non-existent source...")
        try:
            get_dataset_info("non_existent_file_or_dir")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError as e:
            assert "Source not found" in str(e)

    def test_info_missing_images(self):
        print("Testing get_dataset_info on a folder without images/ directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "data.yaml").write_text("names:\n  - apple\n", encoding="utf-8")
            try:
                get_dataset_info(tmpdir)
                assert False, "Should raise FileNotFoundError"
            except FileNotFoundError as e:
                assert "images directory not found" in str(e)
