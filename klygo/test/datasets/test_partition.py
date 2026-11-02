import os
import shutil
import tempfile
import yaml
from pathlib import Path
from klygo.datasets import partition, repartition
from pydantic import ValidationError

def test_partition_and_repartition():
    print("Testing partition and repartition success paths...")
    temp_out = "test_partition_out"
    if os.path.exists(temp_out):
        shutil.rmtree(temp_out)
        
    try:
        # 1. Partition ZIP to folder
        partition(
            source="test/data.zip",
            target=temp_out,
            ratios=(0.8, 0.1, 0.1),
            overwrite=True,
            verbose=False
        )
        
        assert os.path.exists(temp_out)
        assert os.path.exists(os.path.join(temp_out, "images", "train"))
        assert os.path.exists(os.path.join(temp_out, "images", "val"))
        assert os.path.exists(os.path.join(temp_out, "images", "test"))
        
        with open(os.path.join(temp_out, "data.yaml"), "r") as f:
            data = yaml.safe_load(f)
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert data["test"] == "images/test"

        # 2. Repartition folder in-place
        repartition(
            source=temp_out,
            target=temp_out,
            ratios=(0.5, 0.5, 0.0), # 2 splits only
            overwrite=True,
            verbose=False
        )
        
        assert os.path.exists(os.path.join(temp_out, "images", "train"))
        assert os.path.exists(os.path.join(temp_out, "images", "val"))
        assert not os.path.exists(os.path.join(temp_out, "images", "test"))
        
        with open(os.path.join(temp_out, "data.yaml"), "r") as f:
            data2 = yaml.safe_load(f)
        assert data2["train"] == "images/train"
        assert data2["val"] == "images/val"
        assert "test" not in data2

    finally:
        if os.path.exists(temp_out):
            shutil.rmtree(temp_out)

def test_partition_validation_errors():
    print("Testing partition validation errors...")
    # 1. Invalid ratio sum (must equal 1.0 or sum <= 1.0)
    try:
        partition(
            source="test/data.zip",
            target="dummy_out",
            ratios=(0.5, 0.6), # sum = 1.1 > 1.0
            overwrite=True
        )
        assert False, "Should raise ValidationError for ratios sum > 1.0"
    except (ValidationError, ValueError):
        pass

    # 2. Non-existent source directory/file
    try:
        partition(
            source="non_existent_zip_or_folder.zip",
            target="dummy_out",
            ratios=(0.8,),
            overwrite=True
        )
        assert False, "Should raise ValidationError/FileNotFoundError for non-existent source"
    except (ValidationError, FileNotFoundError):
        pass

if __name__ == "__main__":
    test_partition_and_repartition()
    test_partition_validation_errors()
    print("ALL PARTITION TESTS PASSED SUCCESSFULLY!")
