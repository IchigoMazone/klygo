import unittest
import os
import shutil
import tempfile
import time
from klygo.datasets import split
from klygo.archive import list_files
from pydantic import ValidationError

try:
    from ._support import ensure_test_data_zip
except ImportError:
    from test.datasets._support import ensure_test_data_zip

class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_test_data_zip()

    def test_split_success(self):
        print("Testing split by class and ratios...")
        temp_dir = tempfile.mkdtemp(prefix="klygo_split_")

        try:
            # 1. Split by class
            split(
                source="test/data.zip",
                output_dir=temp_dir,
                by_class=True,
                overwrite=True,
                verbose=False
            )
            class_zip = os.path.join(temp_dir, "split_apple")
            assert os.path.exists(class_zip)
            assert os.path.isdir(os.path.join(class_zip, "images"))

            # 2. Split by ratios
            split(
                source="test/data.zip",
                output_dir=temp_dir,
                ratios=(0.5, 0.5),
                overwrite=True,
                verbose=False
            )
            part0 = os.path.join(temp_dir, "split_part_0")
            part1 = os.path.join(temp_dir, "split_part_1")
            assert os.path.exists(part0)
            assert os.path.exists(part1)

        finally:
            for attempt in range(10):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.1)

    def test_split_validation_errors(self):
        print("Testing split validation errors...")
        # Ratios sum > 1.0
        try:
            split(
                source="test/data.zip",
                output_dir="dummy_out",
                ratios=(0.6, 0.5),
                overwrite=True
            )
            assert False, "Should raise ValidationError for ratios sum > 1.0"
        except (ValidationError, ValueError):
            pass

        # Missing split options (neither by_class nor ratios nor class_groups)
        try:
            split(
                source="test/data.zip",
                output_dir="dummy_out",
                overwrite=True
            )
            assert False, "Should raise ValidationError for missing split strategy"
        except (ValidationError, ValueError):
            pass

    def test_split_stratified_ratios_preserves_class_proportions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src")
            os.makedirs(os.path.join(src, "images"), exist_ok=True)
            os.makedirs(os.path.join(src, "labels"), exist_ok=True)
            with open(os.path.join(src, "data.yaml"), "w", encoding="utf-8") as fh:
                fh.write("names:\n  0: majority\n  1: rare\n")

            for i in range(12):
                Image.new("RGB", (16, 16), (i * 15, 0, 0)).save(os.path.join(src, "images", f"img_{i:02d}.png"))
                cls = "0" if i < 8 else "1"
                with open(os.path.join(src, "labels", f"img_{i:02d}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"{cls} 0.5 0.5 0.2 0.2\n")

            out = os.path.join(td, "parts")
            split(
                source=src,
                output_dir=out,
                ratios=(0.5, 0.5),
                stratify=True,
                shuffle=True,
                seed=42,
                overwrite=True,
                verbose=False,
            )

            for part in ("split_part_0", "split_part_1"):
                rare_count = 0
                lbl_dir = os.path.join(out, part, "labels")
                for name in os.listdir(lbl_dir):
                    with open(os.path.join(lbl_dir, name), encoding="utf-8") as fh:
                        if fh.read().startswith("1 "):
                            rare_count += 1
                self.assertEqual(rare_count, 2)

            with self.assertRaises(ValueError):
                split(source=src, output_dir=out, by_class=True, stratify=True, overwrite=True)
