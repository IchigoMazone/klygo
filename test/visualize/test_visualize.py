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
    crop_image,
    read_crops,
)
from klygo.archive import compress
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

        # 3. Test crop_image & read_crops
        crop_dir = tmpdir / "crops"
        cropped_images = crop_image(
            img_path,
            lbl_path,
            classes,
            target=crop_dir,
        )
        assert len(cropped_images) == 2
        assert (crop_dir / "apple" / "sample_crop_0.jpg").exists()

        zip_path = tmpdir / "crops.zip"
        compress(crop_dir, zip_path, verbose=False)
        assert zip_path.exists()
        
        cropped_dict = read_crops(zip_path, grid=None)
        assert "apple" in cropped_dict
        assert "orange" in cropped_dict
        assert len(cropped_dict["apple"]) == 1
        assert len(cropped_dict["orange"]) == 1
        assert isinstance(cropped_dict["apple"][0], Image.Image)
        folder_crops = read_crops(crop_dir, grid=None)
        assert len(folder_crops["apple"]) == 1
        assert len(folder_crops["orange"]) == 1
        default_crop_grid = read_crops(zip_path)
        assert isinstance(default_crop_grid, Image.Image)
        crop_grid = read_crops(crop_dir, grid=(2,))
        assert isinstance(crop_grid, Image.Image)
        fixed_crop_grid = read_crops(zip_path, grid=(1, 1))
        assert isinstance(fixed_crop_grid, Image.Image)
        model_crops = [
            {"image": cropped_images[0], "score": 0.9, "label": "apple"},
            {"image": cropped_images[1], "score": 0.8, "label": "orange"},
        ]
        direct_crops = read_crops(model_crops, grid=None)
        assert set(direct_crops) == {"apple", "orange"}
        assert isinstance(direct_crops["apple"][0], Image.Image)
        direct_grid = read_crops(model_crops)
        assert isinstance(direct_grid, Image.Image)
        direct_metadata = read_crops(model_crops, metadata=True)
        assert direct_metadata[0]["score"] == 0.9
        assert direct_metadata[0]["label"] == "apple"
        folder_metadata = read_crops(crop_dir, metadata=True)
        assert folder_metadata[0]["score"] is None
        assert isinstance(folder_metadata[0]["image"], Image.Image)
        apple_grid = read_crops(model_crops, label="apple")
        assert isinstance(apple_grid, Image.Image)
        apple_group = read_crops(crop_dir, grid=None, label="apple")
        assert set(apple_group) == {"apple"}
        apple_metadata = read_crops(model_crops, metadata=True, label="apple")
        assert len(apple_metadata) == 1
        assert apple_metadata[0]["label"] == "apple"
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
