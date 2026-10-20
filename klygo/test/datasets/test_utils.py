import os
import shutil
import tempfile
import yaml
from pathlib import Path
from klygo.utils.dataset import (
    _safe_copy,
    _safe_move,
    _read_class_names,
    _scan_dataset_files,
    _remap_label_file
)

def test_safe_copy_move():
    print("Testing safe_copy and safe_move...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = tmpdir / "src.txt"
        dst = tmpdir / "dst.txt"
        src.write_text("hello", encoding="utf-8")
        
        # Test copy
        _safe_copy(src, dst)
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "hello"
        assert src.exists()
        
        # Test move
        dst2 = tmpdir / "dst2.txt"
        _safe_move(src, dst2)
        assert dst2.exists()
        assert dst2.read_text(encoding="utf-8") == "hello"
        assert not src.exists()

def test_read_class_names():
    print("Testing _read_class_names...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 1. Missing data.yaml
        try:
            _read_class_names(tmpdir)
            assert False, "Should raise ValueError for missing data.yaml"
        except ValueError as e:
            assert "data.yaml not found" in str(e)
            
        # 2. Missing names key
        yaml_path = tmpdir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"path": "dataset"}, f)
        try:
            _read_class_names(tmpdir)
            assert False, "Should raise ValueError for missing names key"
        except ValueError as e:
            assert "names list not found" in str(e)
            
        # 3. names as list
        with open(yaml_path, "w") as f:
            yaml.dump({"names": ["apple", "banana"]}, f)
        names = _read_class_names(tmpdir)
        assert names == ["apple", "banana"]
        
        # 4. names as dict
        with open(yaml_path, "w") as f:
            yaml.dump({"names": {1: "banana", 0: "apple"}}, f)
        names = _read_class_names(tmpdir)
        assert names == ["apple", "banana"]

def test_scan_dataset_files():
    print("Testing _scan_dataset_files...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        
        # Create some images and labels
        (img_dir / "img1.jpg").write_text("")
        (lbl_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2")
        (img_dir / "img2.png").write_text("")  # No label
        
        # Under splits
        (img_dir / "train").mkdir()
        (lbl_dir / "train").mkdir()
        (img_dir / "train" / "img3.jpg").write_text("")
        (lbl_dir / "train" / "img3.txt").write_text("1 0.1 0.1 0.1 0.1")
        
        pairs = _scan_dataset_files(img_dir, lbl_dir)
        pairs.sort(key=lambda x: str(x[2]))
        
        assert len(pairs) == 3
        # pair 0: img1
        assert pairs[0][0].name == "img1.jpg"
        assert pairs[0][1].name == "img1.txt"
        assert pairs[0][2] == Path("img1.jpg")
        assert pairs[0][3] == Path("img1.txt")
        
        # pair 1: img2 (no label)
        assert pairs[1][0].name == "img2.png"
        assert pairs[1][1] is None
        assert pairs[1][2] == Path("img2.png")
        assert pairs[1][3] == Path("img2.txt")
        
        # pair 2: img3
        assert pairs[2][0].name == "img3.jpg"
        assert pairs[2][1].name == "img3.txt"
        assert pairs[2][2] == Path("train") / "img3.jpg"
        assert pairs[2][3] == Path("train") / "img3.txt"

def test_remap_label_file():
    print("Testing _remap_label_file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = tmpdir / "src.txt"
        dst = tmpdir / "dst.txt"
        
        # Write some bounding boxes
        src.write_text("0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n2 0.9 0.9 0.9 0.9\ninvalid_line\n", encoding="utf-8")
        
        # Class map: 0->2, 1->0
        class_map = {0: 2, 1: 0}
        _remap_label_file(src, dst, class_map)
        
        lines = dst.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        assert lines[0] == "2 0.1 0.2 0.3 0.4"
        assert lines[1] == "0 0.5 0.6 0.7 0.8"
        assert lines[2] == "2 0.9 0.9 0.9 0.9"  # ID 2 not mapped, remains 2
        assert lines[3] == "invalid_line"

if __name__ == "__main__":
    test_safe_copy_move()
    test_read_class_names()
    test_scan_dataset_files()
    test_remap_label_file()
    print("ALL UTILS TESTS PASSED SUCCESSFULLY!")
