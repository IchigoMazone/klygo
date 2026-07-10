import os
import shutil
import tempfile
import yaml
from zipfile import ZipFile
from klygo.datasets import merge
from klygo.archive import extract, list_files
from pydantic import ValidationError

def test_merge_success():
    print("Testing merge success path...")
    temp_dir1 = "test_merge_data1"
    temp_dir2 = "test_merge_data2"
    temp_out_zip = "test_merge_output.zip"
    extracted_out = "test_merge_extracted"

    # Reset
    for d in [temp_dir1, temp_dir2, extracted_out]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(temp_out_zip):
        os.unlink(temp_out_zip)

    try:
        os.makedirs(os.path.join(temp_dir1, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir1, "labels"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir2, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir2, "labels"), exist_ok=True)

        # Write YAML data
        with open(os.path.join(temp_dir1, "data.yaml"), "w") as f:
            yaml.dump({"names": ["apple", "banana"]}, f)
        with open(os.path.join(temp_dir2, "data.yaml"), "w") as f:
            yaml.dump({"names": ["banana", "orange"]}, f)

        # Write dummy images and labels
        with open(os.path.join(temp_dir1, "images", "img1.jpg"), "w") as f: f.write("")
        with open(os.path.join(temp_dir1, "labels", "img1.txt"), "w") as f: f.write("0 0.5 0.5 0.2 0.2\n1 0.6 0.6 0.2 0.2")

        with open(os.path.join(temp_dir2, "images", "img2.jpg"), "w") as f: f.write("")
        with open(os.path.join(temp_dir2, "labels", "img2.txt"), "w") as f: f.write("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.2 0.2")

        # Merge them
        merge(
            sources=[temp_dir1, temp_dir2],
            output=temp_out_zip,
            overwrite=True,
            verbose=False
        )

        assert os.path.exists(temp_out_zip)
        extract(temp_out_zip, extracted_out)

        # Read global yaml
        with open(os.path.join(extracted_out, "data.yaml"), "r") as f:
            gdata = yaml.safe_load(f)
        
        # Global names list should be unique combined: apple, banana, orange
        assert gdata["names"] == ["apple", "banana", "orange"]
        assert gdata["nc"] == 3

        # Verify class ID remapping
        # temp_dir1: apple->0, banana->1. Global: apple->0, banana->1. No remapping needed for source 1.
        with open(os.path.join(extracted_out, "labels", "src0_img1.txt"), "r") as f:
            lines = f.readlines()
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")

        # temp_dir2: banana->0, orange->1. Global: banana->1, orange->2. Banana mapped from 0 to 1, Orange from 1 to 2.
        with open(os.path.join(extracted_out, "labels", "src1_img2.txt"), "r") as f:
            lines = f.readlines()
        assert lines[0].startswith("1 ")
        assert lines[1].startswith("2 ")

    finally:
        for d in [temp_dir1, temp_dir2, extracted_out]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(temp_out_zip):
            os.unlink(temp_out_zip)

def test_merge_validation_errors():
    print("Testing merge validation errors...")
    # Non-existent source
    try:
        merge(
            sources=["non_existent_source.zip", "test/data.zip"],
            output="dummy.zip",
            overwrite=True
        )
        assert False, "Should raise FileNotFoundError for non-existent source"
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    test_merge_success()
    test_merge_validation_errors()
    print("ALL MERGE TESTS PASSED SUCCESSFULLY!")
