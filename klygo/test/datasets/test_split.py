import os
import shutil
from klygo.datasets import split
from klygo.archive import list_files
from pydantic import ValidationError

def test_split_success():
    print("Testing split by class and ratios...")
    temp_dir = "test_split_out"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

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
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_split_validation_errors():
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

if __name__ == "__main__":
    test_split_success()
    test_split_validation_errors()
    print("ALL SPLIT TESTS PASSED SUCCESSFULLY!")
