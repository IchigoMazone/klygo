import unittest
import os
import shutil
import tempfile
import time
from klygo.datasets import split
from klygo.archive import list_files
from pydantic import ValidationError

class TestSplit(unittest.TestCase):
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
            class_zip = os.path.join(temp_dir, "split_apple.zip")
            assert os.path.exists(class_zip)
            assert len(list_files(class_zip)) > 0

            # 2. Split by ratios
            split(
                source="test/data.zip",
                output_dir=temp_dir,
                ratios=(0.5, 0.5),
                overwrite=True,
                verbose=False
            )
            part0 = os.path.join(temp_dir, "split_part_0.zip")
            part1 = os.path.join(temp_dir, "split_part_1.zip")
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
