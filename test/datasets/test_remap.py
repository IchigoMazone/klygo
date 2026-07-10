import os
import shutil
import yaml
import time
from klygo.datasets import remap_classes
from klygo.archive import extract
from pydantic import ValidationError


def safe_rmtree(path):
    if not os.path.exists(path):
        return
    for _ in range(10):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.1)
    shutil.rmtree(path)


def safe_unlink(path):
    if not os.path.exists(path):
        return
    for _ in range(10):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.1)
    os.unlink(path)


def test_remap_success_list_format():
    print("Testing remap_classes with continuous list format...")
    temp_zip = "test_remap_out.zip"
    extracted_out = "test_remap_extracted"
    
    safe_unlink(temp_zip)
    safe_rmtree(extracted_out)

    try:
        # Renaming class "apple" to "fresh_apple", class ID remains 0 (continuous)
        remap_classes(
            source="test/data.zip",
            output=temp_zip,
            class_map={"apple": "fresh_apple"},
            overwrite=True,
            verbose=False
        )

        assert os.path.exists(temp_zip)
        extract(temp_zip, extracted_out)

        with open(os.path.join(extracted_out, "data.yaml"), "r") as f:
            data = yaml.safe_load(f)
        
        # Continuous ID -> names should be saved as list!
        assert data["names"] == ["fresh_apple"]
        assert data["nc"] == 1

    finally:
        safe_unlink(temp_zip)
        safe_rmtree(extracted_out)

def test_remap_success_dict_format():
    print("Testing remap_classes with sparse dict format...")
    temp_zip = "test_remap_out.zip"
    extracted_out = "test_remap_extracted"
    
    safe_unlink(temp_zip)
    safe_rmtree(extracted_out)

    try:
        # Changing ID from 0 to 1, rename "apple" to "red_apple" (sparse)
        remap_classes(
            source="test/data.zip",
            output=temp_zip,
            class_map={0: 1, "apple": "red_apple"},
            overwrite=True,
            verbose=False
        )

        assert os.path.exists(temp_zip)
        extract(temp_zip, extracted_out)

        with open(os.path.join(extracted_out, "data.yaml"), "r") as f:
            data = yaml.safe_load(f)
        
        # Sparse ID -> names should be saved as dict!
        assert data["names"] == {1: "red_apple"}
        assert data["nc"] == 2

    finally:
        safe_unlink(temp_zip)
        safe_rmtree(extracted_out)

def test_remap_validation_errors():
    print("Testing remap_classes validation errors...")
    # Mapping non-existent class
    try:
        remap_classes(
            source="test/data.zip",
            output="dummy.zip",
            class_map={"non_existent_class": "orange"},
            overwrite=True
        )
        assert False, "Should raise ValueError for non-existent class"
    except ValueError as e:
        assert "not found in source dataset" in str(e)

    # Mapping out of bounds class ID
    try:
        remap_classes(
            source="test/data.zip",
            output="dummy.zip",
            class_map={5: "orange"},
            overwrite=True
        )
        assert False, "Should raise ValueError for out of range ID"
    except ValueError as e:
        assert "out of range of source dataset classes" in str(e)

    # Invalid type for mapping key
    try:
        remap_classes(
            source="test/data.zip",
            output="dummy.zip",
            class_map={1.5: "orange"}, # float key
            overwrite=True
        )
        assert False, "Should raise TypeError for float key"
    except (TypeError, ValidationError):
        pass

if __name__ == "__main__":
    test_remap_success_list_format()
    test_remap_success_dict_format()
    test_remap_validation_errors()
    print("ALL REMAP TESTS PASSED SUCCESSFULLY!")
