import os
import shutil
import tempfile
from pathlib import Path
from klygo.datasets import get_dataset_info
from klygo.archive import extract

def test_info_valid_zip():
    print("Testing get_dataset_info on a valid ZIP file...")
    info = get_dataset_info("test/data.zip")
    assert info["nc"] == 1
    assert info["classes"] == ["apple"]
    assert info["total_images"] == 361
    assert info["total_labels"] == 361
    assert info["unlabeled_images"] == 0
    assert "raw" in info["splits"]
    assert info["splits"]["raw"] == 361

def test_info_valid_folder():
    print("Testing get_dataset_info on a valid folder...")
    with tempfile.TemporaryDirectory() as tmpdir:
        extract("test/data.zip", tmpdir)
        info = get_dataset_info(tmpdir)
        assert info["nc"] == 1
        assert info["classes"] == ["apple"]
        assert info["total_images"] == 361
        assert info["total_labels"] == 361
        assert info["unlabeled_images"] == 0
        assert "raw" in info["splits"]

def test_info_nonexistent():
    print("Testing get_dataset_info on a non-existent source...")
    try:
        get_dataset_info("non_existent_file_or_dir")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as e:
        assert "Source not found" in str(e)

def test_info_missing_images():
    print("Testing get_dataset_info on a folder without images/ directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "data.yaml").write_text("names:\n  - apple\n", encoding="utf-8")
        try:
            get_dataset_info(tmpdir)
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError as e:
            assert "images directory not found" in str(e)

if __name__ == "__main__":
    test_info_valid_zip()
    test_info_valid_folder()
    test_info_nonexistent()
    test_info_missing_images()
    print("ALL INFO TESTS PASSED SUCCESSFULLY!")
