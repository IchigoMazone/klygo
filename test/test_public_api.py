from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from klygo.archive import (
    add,
    compress,
    extract,
    extract_file,
    get_info,
    human_size,
    list_files,
    merge as merge_archives,
    remove,
    search,
    split_by_size,
    test as test_archive,
)
from klygo.models import Model
from klygo.io import (
    Config,
    read_images,
    read_json,
    read_toml,
    read_yaml,
    write_json,
    write_toml,
    write_yaml,
)
from klygo.visualize import (
    crop_dataset,
    crop_image,
    draw_bboxes,
    plot_dataset_stats,
    read_crops,
    read_detections,
    show_image,
    visualize_dataset_image,
    visualize_prediction,
)


def test_all_archive_public_functions(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "one.txt").write_text("one", encoding="utf-8")
    (source_dir / "two.txt").write_text("two", encoding="utf-8")

    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    compress(source_dir, first_zip, verbose=False)
    compress(source_dir / "one.txt", second_zip, verbose=False)

    names = list_files(first_zip)
    assert len(names) == 2
    assert get_info(first_zip)["file_count"] == 2
    assert test_archive(first_zip) is True
    assert search(first_zip, "*.txt")

    extracted_dir = tmp_path / "extracted"
    extract(first_zip, extracted_dir, verbose=False)
    assert any(extracted_dir.rglob("one.txt"))

    single_dir = tmp_path / "single"
    extract_file(first_zip, names[0], single_dir)
    assert (single_dir / Path(names[0]).name).exists()

    extra_path = tmp_path / "extra.txt"
    extra_path.write_text("extra", encoding="utf-8")
    add(first_zip, extra_path, verbose=False)
    assert "extra.txt" in list_files(first_zip)
    remove(first_zip, "extra.txt")
    assert "extra.txt" not in list_files(first_zip)

    merged_path = tmp_path / "merged.zip"
    merge_archives([first_zip, second_zip], merged_path, verbose=False)
    assert merged_path.exists()

    parts = split_by_size(
        merged_path,
        size=0.000001,
        output_dir=tmp_path / "parts",
        overwrite=True,
        verbose=False,
    )
    assert parts
    assert human_size(0) == "0.00 B"
    assert human_size(1024) == "1.00 KB"


def test_all_visualize_public_functions(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    image = Image.fromarray(np.full((100, 120, 3), 255, dtype=np.uint8))
    boxes = [[20, 20, 80, 80]]
    labels = ["apple"]
    scores = [0.9]

    annotated = draw_bboxes(
        image,
        boxes,
        labels,
        scores,
        box_color=(0, 255, 0),
        text_color=(0, 0, 255),
        font=cv.FONT_HERSHEY_DUPLEX,
    )
    assert isinstance(annotated, Image.Image)
    show_image(annotated)
    visualize_prediction(
        image,
        {"boxes": boxes, "labels": labels, "scores": scores},
    )

    image_path = tmp_path / "image.jpg"
    label_path = tmp_path / "label.txt"
    image.save(image_path)
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    visualize_dataset_image(image_path, label_path, labels)

    crop_dir = tmp_path / "single_crops"
    image_crops = crop_image(
        image_path,
        label_path,
        labels,
        target=crop_dir,
        padding=2,
    )
    assert len(image_crops) == 1
    assert isinstance(image_crops[0], Image.Image)
    assert (crop_dir / "apple" / "image_crop_0.jpg").exists()

    crop_path = tmp_path / "crops.zip"
    compress(crop_dir, crop_path, verbose=False)
    crops = read_crops(crop_path, grid=None)
    assert len(crops["apple"]) == 1
    folder_crops = read_crops(crop_dir, grid=None)
    assert len(folder_crops["apple"]) == 1
    default_crop_grid = read_crops(crop_path)
    assert isinstance(default_crop_grid, Image.Image)
    crop_grid = read_crops(crop_dir, grid=(1,))
    assert isinstance(crop_grid, Image.Image)
    fixed_crop_grid = read_crops(crop_path, grid=(1, 1))
    assert isinstance(fixed_crop_grid, Image.Image)

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "labels").mkdir()
    image.save(dataset_dir / "images" / "image.jpg")
    (dataset_dir / "labels" / "image.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (dataset_dir / "data.yaml").write_text(
        "names: [apple]\n",
        encoding="utf-8",
    )
    plot_dataset_stats(dataset_dir)

    flat_crops = crop_dataset(dataset_dir, target=tmp_path / "flat_crops")
    assert len(flat_crops) == 1
    assert isinstance(flat_crops[0], Image.Image)
    assert (tmp_path / "flat_crops" / "apple" / "image_crop_0.jpg").exists()

    split_dataset = tmp_path / "split_dataset"
    for split_name in ("train", "val", "test"):
        (split_dataset / "images" / split_name).mkdir(parents=True)
        (split_dataset / "labels" / split_name).mkdir(parents=True)
        image.save(split_dataset / "images" / split_name / f"{split_name}.jpg")
        (split_dataset / "labels" / split_name / f"{split_name}.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )
    (split_dataset / "data.yaml").write_text(
        "names: [apple]\n",
        encoding="utf-8",
    )
    split_crops = crop_dataset(split_dataset)
    assert len(split_crops) == 3
    assert all(isinstance(crop, Image.Image) for crop in split_crops)


class _Inputs(dict):
    input_ids = torch.tensor([[1]])

    def to(self, device):
        return self


class _Processor:
    def __call__(self, **kwargs):
        return _Inputs()

    def post_process_grounded_object_detection(self, **kwargs):
        return [{
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 90.0, 90.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": ["small", "large"],
        }]


def test_model_predict_without_downloading_model(tmp_path):
    model = object.__new__(Model)
    model.device = "cpu"
    model.processor = _Processor()
    model.model = lambda **kwargs: SimpleNamespace()
    model.input_kwargs = {}
    model.output_kwargs = {}

    result = model.predict(
        Image.new("RGB", (100, 100)),
        box_threshold=0.2,
        max_area=0.2,
    )[0]
    assert result["labels"] == ["small"]
    assert len(result["boxes"]) == len(result["scores"]) == 1

    image_dir = tmp_path / "predict_images"
    image_dir.mkdir()
    Image.new("RGB", (100, 100), "white").save(image_dir / "image.jpg")
    directory_result = model.predict(source=image_dir, max_area=0.2)[0]
    assert directory_result["labels"] == ["small"]
    opencv_images = read_images(image_dir, backend="opencv")
    opencv_result = model.predict(source=opencv_images, max_area=0.2)[0]
    assert opencv_result["labels"] == ["small"]


def test_model_init_and_setters_without_downloading_model(monkeypatch):
    monkeypatch.setattr(Model, "_model", lambda self, model: "model")
    monkeypatch.setattr(Model, "_processor", lambda self, model: "processor")
    model = Model()
    assert model.model == "model"
    assert model.processor == "processor"
    model.set_input_kwargs(padding=True)
    model.set_output_kwargs(top_k=5)
    assert model.input_kwargs == {"padding": True}
    assert model.output_kwargs == {"top_k": 5}


def test_model_private_loaders_without_downloading_model(monkeypatch):
    import klygo.models.model as model_module

    class FakeLoadedModel:
        def to(self, device):
            self.device = device
            return self

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(name):
            return {"processor": name}

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(name):
            return FakeLoadedModel()

    monkeypatch.setattr(model_module, "AutoProcessor", FakeProcessorFactory)
    monkeypatch.setattr(
        model_module,
        "AutoModelForZeroShotObjectDetection",
        FakeModelFactory,
    )
    model = object.__new__(Model)
    model.device = "cpu"
    model.name_models = Model.NAME_MODELS
    model.model_hugging_face = Model.MODEL_HUGGING_FACE
    assert model._processor(Model.NAME_MODELS["GDN"])["processor"]
    assert model._model(Model.NAME_MODELS["GDN"]).device == "cpu"


def test_config_private_path_expansion(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "default:\n  root: /data\nsection:\n  nested:\n    path: ./images\n",
        encoding="utf-8",
    )
    config = Config(config_path)
    result = config.read(verbose=False)
    assert result.section.nested.path == "/data/images"


def test_all_io_format_public_functions(tmp_path):
    data = {"name": "apple", "count": 2}
    yaml_path = tmp_path / "data.yaml"
    json_path = tmp_path / "data.json"
    toml_path = tmp_path / "data.toml"

    write_yaml(yaml_path, data, overwrite=True, verbose=False)
    write_json(json_path, data, overwrite=True, verbose=False)
    write_toml(toml_path, data, overwrite=True, verbose=False)
    assert dict(read_yaml(yaml_path, verbose=False)) == data
    assert read_json(json_path, verbose=False) == data
    assert read_toml(toml_path, verbose=False) == data

    config = Config(yaml_path)
    assert config.config_path == yaml_path


def test_read_images_with_pil_and_opencv_backends(tmp_path):
    image_dir = tmp_path / "images"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (12, 8), (255, 0, 0)).save(image_dir / "red.jpg")
    Image.new("RGB", (6, 4), (0, 255, 0)).save(nested_dir / "green.png")

    pil_images = read_images(image_dir)
    assert len(pil_images) == 1
    assert isinstance(pil_images[0], Image.Image)
    assert pil_images[0].mode == "RGB"

    recursive_images = read_images(image_dir, recursive=True)
    assert len(recursive_images) == 2

    opencv_images = read_images(image_dir / "red.jpg", backend="opencv")
    assert len(opencv_images) == 1
    assert isinstance(opencv_images[0], np.ndarray)
    assert opencv_images[0].shape == (8, 12, 3)
    assert opencv_images[0][0, 0, 2] > opencv_images[0][0, 0, 0]


def test_model_detect_without_downloading_model(tmp_path, monkeypatch):
    import klygo.models.model as model_module
    import klygo.utils.media as media_module

    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    class FakeCapture:
        reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv.CAP_PROP_FPS: 1,
                cv.CAP_PROP_FRAME_WIDTH: 64,
                cv.CAP_PROP_FRAME_HEIGHT: 48,
                cv.CAP_PROP_FRAME_COUNT: 1,
            }[prop]

        def read(self):
            if self.reads:
                return False, None
            self.reads += 1
            return True, frame.copy()

        def release(self):
            return None

    monkeypatch.setattr(media_module.cv, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(model_module, "tqdm", lambda iterable, **kwargs: iterable)

    model = object.__new__(Model)
    model.predict = MethodType(
        lambda self, **kwargs: [{
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": ["apple"],
        }],
        model,
    )
    model._imwrite_txt = MethodType(Model._imwrite_txt, model)

    video_path = tmp_path / "input.mp4"
    video_path.touch()
    detected_images = model.detect(
        str(video_path),
        annotated_dir=str(tmp_path / "annotated"),
        dataset_dir=str(tmp_path / "dataset"),
    )
    assert (tmp_path / "annotated" / "frame_0.jpg").exists()
    assert (tmp_path / "dataset" / "images" / "frame_0.jpg").exists()
    assert (tmp_path / "dataset" / "labels" / "frame_0.txt").exists()
    assert (tmp_path / "dataset" / "data.yaml").exists()
    dataset_yaml = read_yaml(
        tmp_path / "dataset" / "data.yaml",
        verbose=False,
    )
    assert dataset_yaml["train"] == "images"
    assert "val" not in dataset_yaml
    assert "test" not in dataset_yaml
    assert len(detected_images) == 1
    assert isinstance(detected_images[0], Image.Image)
    assert (tmp_path / "annotated" / "metadata.json").exists()
    saved_detections = read_detections(tmp_path / "annotated")
    assert len(saved_detections) == 1
    assert isinstance(saved_detections[0]["image"], Image.Image)
    assert saved_detections[0]["boxes"] == [[5.0, 5.0, 20.0, 20.0]]
    assert abs(saved_detections[0]["scores"][0] - 0.9) < 1e-6
    assert saved_detections[0]["labels"] == ["apple"]
    detection_zip = tmp_path / "detections.zip"
    compress(tmp_path / "annotated", detection_zip, verbose=False)
    zipped_detections = read_detections(detection_zip)
    assert zipped_detections[0]["labels"] == ["apple"]
    legacy_detections = tmp_path / "legacy_detections"
    legacy_detections.mkdir()
    Image.new("RGB", (16, 12), "white").save(
        legacy_detections / "frame_0.jpg"
    )
    legacy_records = read_detections(legacy_detections)
    assert len(legacy_records) == 1
    assert legacy_records[0]["boxes"] == []
    assert legacy_records[0]["scores"] == []
    assert legacy_records[0]["labels"] == []

    image_dir = tmp_path / "source_images"
    image_dir.mkdir()
    Image.new("RGB", (64, 48), "white").save(image_dir / "image.jpg")
    source_images = read_images(image_dir)
    image_detections = model.detect(
        source_images,
        annotated_dir=str(tmp_path / "image_annotated"),
    )
    assert len(image_detections) == 1
    assert isinstance(image_detections[0], Image.Image)
    direct_detections = model.detect(source_images, metadata=True)
    assert len(direct_detections) == 1
    assert isinstance(direct_detections[0]["image"], Image.Image)
    assert direct_detections[0]["boxes"] == [[5.0, 5.0, 20.0, 20.0]]
    assert direct_detections[0]["labels"] == ["apple"]
    normalized_detections = read_detections(direct_detections)
    assert normalized_detections[0]["labels"] == ["apple"]

    crop_results = model.crop(
        source=str(video_path),
        target=str(tmp_path / "crops"),
        padding=2,
    )
    assert len(crop_results) == 1
    assert isinstance(crop_results[0]["image"], Image.Image)
    assert abs(crop_results[0]["score"] - 0.9) < 1e-6
    assert crop_results[0]["label"] == "apple"
    assert (tmp_path / "crops" / "metadata.json").exists()
    saved_results = read_crops(tmp_path / "crops", metadata=True)
    assert len(saved_results) == 1
    assert isinstance(saved_results[0]["image"], Image.Image)
    assert abs(saved_results[0]["score"] - 0.9) < 1e-6
    assert saved_results[0]["label"] == "apple"
    saved_zip = tmp_path / "model_crops.zip"
    compress(tmp_path / "crops", saved_zip, verbose=False)
    zipped_results = read_crops(saved_zip, metadata=True)
    assert abs(zipped_results[0]["score"] - 0.9) < 1e-6
    assert zipped_results[0]["label"] == "apple"
    with pytest.raises(FileExistsError):
        model.crop(source=str(video_path), target=str(tmp_path / "crops"))
    overwritten_results = model.crop(
        source=str(video_path),
        target=str(tmp_path / "crops"),
        overwrite=True,
    )
    assert len(overwritten_results) == 1
    assert not (tmp_path / "crops" / ".metadata.json.tmp").exists()

    directory_crops = model.crop(source=image_dir, padding=2)
    assert len(directory_crops) == 1
    assert isinstance(directory_crops[0]["image"], Image.Image)
    saved_crop = tmp_path / "crops" / "apple" / "frame_0_crop_0.jpg"
    assert saved_crop.exists()

    crop_results = model.crop(source=str(video_path), padding=2)
    assert len(crop_results) == 1
    assert isinstance(crop_results[0]["image"], Image.Image)
    assert crop_results[0]["image"].size == (19, 19)
    assert abs(crop_results[0]["score"] - 0.9) < 1e-6
    assert crop_results[0]["label"] == "apple"
