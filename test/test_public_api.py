from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
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
    split as split_archive,
    test as test_archive,
)
from klygo.models import Kernel, Model
from klygo.io import (
    Config,
    read_json,
    read_toml,
    read_yaml,
    write_json,
    write_toml,
    write_yaml,
)
from klygo.visualize import (
    crop_dataset,
    crop_objects,
    draw_bboxes,
    plot_dataset_stats,
    read_cropped_objects,
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

    parts = split_archive(
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

    crop_path = tmp_path / "crops.zip"
    crop_objects(image_path, label_path, labels, crop_path)
    crops = read_cropped_objects(crop_path)
    assert len(crops["apple"]) == 1

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


def test_kernel_predict_without_downloading_model():
    kernel = object.__new__(Kernel)
    kernel.device = "cpu"
    kernel.processor = _Processor()
    kernel.model = lambda **kwargs: SimpleNamespace()
    kernel.input_kwargs = {}
    kernel.output_kwargs = {}

    result = kernel.predict(
        Image.new("RGB", (100, 100)),
        box_threshold=0.2,
        max_area=0.2,
    )[0]
    assert result["labels"] == ["small"]
    assert len(result["boxes"]) == len(result["scores"]) == 1


def test_kernel_init_and_setters_without_downloading_model(monkeypatch):
    monkeypatch.setattr(Kernel, "_model", lambda self, model: "model")
    monkeypatch.setattr(Kernel, "_processor", lambda self, model: "processor")
    kernel = Kernel()
    assert kernel.model == "model"
    assert kernel.processor == "processor"
    kernel.set_input_kwargs(padding=True)
    kernel.set_output_kwargs(top_k=5)
    assert kernel.input_kwargs == {"padding": True}
    assert kernel.output_kwargs == {"top_k": 5}
    assert Kernel is Model


def test_kernel_private_loaders_without_downloading_model(monkeypatch):
    import klygo.models.kernel as kernel_module

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

    monkeypatch.setattr(kernel_module, "AutoProcessor", FakeProcessorFactory)
    monkeypatch.setattr(
        kernel_module,
        "AutoModelForZeroShotObjectDetection",
        FakeModelFactory,
    )
    kernel = object.__new__(Kernel)
    kernel.device = "cpu"
    kernel.name_models = Kernel.NAME_MODELS
    kernel.model_hugging_face = Kernel.MODEL_HUGGING_FACE
    assert kernel._processor(Kernel.NAME_MODELS["GDN"])["processor"]
    assert kernel._model(Kernel.NAME_MODELS["GDN"]).device == "cpu"


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


def test_kernel_detect_without_downloading_model(tmp_path, monkeypatch):
    import klygo.models.kernel as kernel_module

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

    monkeypatch.setattr(kernel_module.cv, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(kernel_module, "tqdm", lambda iterable, **kwargs: iterable)

    kernel = object.__new__(Kernel)
    kernel.predict = MethodType(
        lambda self, **kwargs: [{
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": ["apple"],
        }],
        kernel,
    )
    kernel._imwrite_txt = MethodType(Kernel._imwrite_txt, kernel)

    video_path = tmp_path / "input.mp4"
    video_path.touch()
    detected_images = kernel.detect(
        str(video_path),
        annotated_dir=str(tmp_path / "annotated"),
        dataset_dir=str(tmp_path / "dataset"),
    )
    assert (tmp_path / "annotated" / "frame_0.jpg").exists()
    assert (tmp_path / "dataset" / "images" / "frame_0.jpg").exists()
    assert (tmp_path / "dataset" / "labels" / "frame_0.txt").exists()
    assert (tmp_path / "dataset" / "data.yaml").exists()
    assert len(detected_images) == 1
    assert isinstance(detected_images[0], Image.Image)

    crop_images = kernel.crop(
        input_path=str(video_path),
        target=str(tmp_path / "crops"),
        padding=2,
    )
    assert len(crop_images) == 1
    assert isinstance(crop_images[0], Image.Image)
    saved_crop = tmp_path / "crops" / "apple" / "frame_0_crop_0.jpg"
    assert saved_crop.exists()

    crop_images = kernel.crop(input_path=str(video_path), padding=2)
    assert len(crop_images) == 1
    assert isinstance(crop_images[0], Image.Image)
    assert crop_images[0].size == (19, 19)
