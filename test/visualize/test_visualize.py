import sys
import pathlib
import tempfile
import shutil
from PIL import Image
import numpy as np

# Thêm root dự án vào sys.path để python nhận diện package klygo local
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from klygo.visualize import (
    draw_bboxes,
    visualize_prediction,
    visualize_dataset_image,
    crop_objects,
    read_cropped_objects,
)
from klygo.datasets import partition, unpartition

def test_visualization_and_crop():
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    try:
        # 1. Tạo dữ liệu giả
        classes = ["apple", "orange"]
        img = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
        
        img_path = tmpdir / "sample.jpg"
        lbl_path = tmpdir / "sample.txt"
        img.save(img_path)
        
        # Nhãn YOLO: class_id, x_center, y_center, w, h
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n") # apple ở giữa
            f.write("1 0.2 0.2 0.1 0.1\n") # orange ở góc

        # 2. Test draw_bboxes & visualize_prediction
        pred = {
            "boxes": [[100, 100, 200, 200]],
            "scores": [0.89],
            "labels": ["apple"],
        }
        # Chỉ test vẽ không raise lỗi (vì pyplot hiển thị đòi hỏi GUI/màn hình)
        annotated = draw_bboxes(img, pred["boxes"], pred["labels"], pred["scores"])
        assert isinstance(annotated, Image.Image)

        # 3. Test crop_objects & read_cropped_objects
        zip_path = tmpdir / "crops.zip"
        crop_objects(img_path, lbl_path, classes, zip_path, overwrite=True)
        assert zip_path.exists()
        
        cropped_dict = read_cropped_objects(zip_path)
        assert "apple" in cropped_dict
        assert "orange" in cropped_dict
        assert len(cropped_dict["apple"]) == 1
        assert len(cropped_dict["orange"]) == 1
        assert isinstance(cropped_dict["apple"][0], Image.Image)
        print("Crop and Read Crop: OK!")

        # 4. Test unpartition
        # Tạo cấu trúc dataset giả đã split
        split_ds = tmpdir / "split_dataset"
        os_makedirs = lambda p: p.mkdir(parents=True, exist_ok=True)
        os_makedirs(split_ds / "images" / "train")
        os_makedirs(split_ds / "labels" / "train")
        
        shutil.copy(img_path, split_ds / "images" / "train" / "sample.jpg")
        shutil.copy(lbl_path, split_ds / "labels" / "train" / "sample.txt")
        
        with open(split_ds / "data.yaml", "w") as f:
            f.write("names: ['apple', 'orange']\n")

        flat_ds = tmpdir / "flat_dataset"
        unpartition(split_ds, flat_ds, overwrite=True, verbose=False)
        
        assert (flat_ds / "images" / "sample.jpg").exists()
        assert (flat_ds / "labels" / "sample.txt").exists()
        assert (flat_ds / "data.yaml").exists()
        print("Unpartition: OK!")

    finally:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    test_visualization_and_crop()
    print("ALL VISUALIZATION AND CROP TESTS PASSED!")
