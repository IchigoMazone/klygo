from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from klygo import media, processing as p
from klygo.outputs.detect import Box, Detection, Detections


def _image_path(tmp_path: Path, size=(100, 50)) -> Path:
    path = tmp_path / "input.png"
    Image.new("RGB", size, (120, 80, 40)).save(path)
    return path


def test_compose_is_lazy_and_materializes_once(tmp_path):
    image = media.load(_image_path(tmp_path))[0]
    pipeline = p.compose([
        p.crop((10, 5, 90, 45)),
        p.resize((40, 20)),
        p.clahe(),
    ])

    processed = pipeline(image)

    assert isinstance(processed, media.LazyImage)
    assert not processed.is_loaded
    assert processed.size == (40, 20)
    assert p.original_size(processed) == (100, 50)
    assert p.processed_size(processed) == (40, 20)
    assert len(p.inspect(processed)) == 3

    output = p.materialize(processed, output="array", cache=False)
    assert output.shape == (20, 40, 3)
    assert not processed.is_loaded
    assert [item["name"] for item in p.get_trace(processed)] == ["crop", "resize"]


def test_compose_supports_list_and_variadic_syntax():
    first = p.compose([p.grayscale(), p.threshold(127)])
    second = p.compose(p.grayscale(), p.threshold(127))
    assert [item["name"] for item in p.inspect(first)] == ["grayscale", "threshold"]
    assert p.inspect(first) == p.inspect(second)


def test_restore_boxes_after_letterbox(tmp_path):
    image = media.load(_image_path(tmp_path, size=(100, 50)))[0]
    processed = p.letterbox((100, 100))(image)
    p.materialize(processed, output="array", cache=False)

    restored = p.restore_boxes([[0, 25, 100, 75]], p.get_trace(processed))
    assert np.allclose(restored[0], [0, 0, 100, 50])


def test_detection_restore_supports_transformed_and_original_spaces(tmp_path):
    image = media.load(_image_path(tmp_path, size=(100, 50)))[0]
    processed = p.compose([p.resize((40, 20)), p.grayscale()])(image)
    p.materialize(processed, output="array", cache=False)
    detection = Detection(
        source_image=processed,
        objects=[Box(0, "object", 0.9, [10, 5, 30, 15])],
    )

    transformed = detection.restore(target="transformed")
    assert transformed is not detection
    assert transformed.source_image.size == (40, 20)
    assert transformed.source_image.to_pil().mode == "RGB"
    assert transformed.boxes == [[10.0, 5.0, 30.0, 15.0]]

    original = detection.restore(target="original")
    assert original.source_image.size == (100, 50)
    assert original.source_image.to_pil().mode == "RGB"
    assert np.allclose(original.boxes[0], [25, 12.5, 75, 37.5])
    assert detection.source_image.size == (40, 20)


def test_detection_restore_can_update_inplace(tmp_path):
    image = media.load(_image_path(tmp_path, size=(100, 50)))[0]
    processed = p.resize((50, 25))(image)
    p.materialize(processed, output="array", cache=False)
    detection = Detection(processed, [Box(0, "object", 0.8, [5, 5, 25, 20])])

    returned = detection.restore(target="original", inplace=True)

    assert returned is detection
    assert detection.source_image.size == (100, 50)
    assert np.allclose(detection.boxes[0], [10, 10, 50, 40])
    assert detection.objects[0].parent_image is detection.source_image

    restored_collection = Detections([detection]).restore(target="transformed")
    assert isinstance(restored_collection, Detections)
    assert restored_collection[0].source_image.size == (100, 50)


def test_metrics_and_predicates_accept_lazy_image(tmp_path):
    image = media.load(_image_path(tmp_path))[0]
    condition = p.all_of(
        p.brightness_score() > 0,
        p.contrast_score() >= 0,
        p.entropy_score() >= 0,
    )
    assert condition(image)
    assert not image.is_loaded


def test_stream_transform_remains_lazy(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for index in range(3):
        Image.new("RGB", (20, 10), (index * 20, 50, 100)).save(directory / f"{index}.png")

    stream = media.stream(directory).transform(p.compose([p.resize((8, 8)), p.grayscale()]))
    frames = list(stream)

    assert len(frames) == 3
    assert all(isinstance(frame, media.LazyImage) for frame in frames)
    assert all(frame.size == (8, 8) for frame in frames)
    assert all(not frame.is_loaded for frame in frames)


def test_advanced_processing_groups_execute(tmp_path):
    image = media.load(_image_path(tmp_path))[0]
    pipeline = p.compose([
        p.autocontrast(),
        p.gaussian_blur(3),
        p.otsu_threshold(),
        p.opening(3),
        p.canny(50, 100),
        p.cast("float32"),
        p.rescale(1 / 255),
    ])
    result = p.materialize(pipeline(image), output="array", cache=False)
    assert result.shape == (50, 100)
    assert result.dtype == np.float32


@pytest.mark.parametrize("operation", [
    p.center_crop((40, 30)), p.thumbnail((40, 40)), p.contain((40, 40)),
    p.cover((40, 40)), p.fit((40, 40)), p.pad(2), p.flip(True),
    p.rotate(5), p.transpose("rotate_90"), p.affine((1, 0, 0, 0, 1, 0)),
    p.perspective((1, 0, 0, 0, 1, 0, 0, 0)), p.auto_orient(), p.strip_exif(),
    p.convert_color("HSV"), p.grayscale(), p.add_alpha(), p.remove_alpha(),
    p.select_channels((0, 1)), p.swap_channels((2, 1, 0)), p.brightness(1.1),
    p.contrast(1.1), p.saturation(1.1), p.hue(0.1), p.gamma(1.1),
    p.temperature(0.1), p.tint(0.1), p.white_balance(), p.autocontrast(),
    p.equalize(), p.clahe(), p.invert(), p.solarize(), p.posterize(4),
    p.colorize("black", "white"), p.box_blur(), p.gaussian_blur(),
    p.median_blur(), p.bilateral_filter(), p.sharpen(), p.unsharp_mask(),
    p.denoise(), p.detail(), p.smooth(), p.emboss(),
    p.convolve([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    p.threshold(127), p.adaptive_threshold(), p.otsu_threshold(),
    p.triangle_threshold(), p.erode(), p.dilate(), p.opening(), p.closing(),
    p.morphological_gradient(), p.tophat(), p.blackhat(), p.canny(), p.sobel(),
    p.scharr(), p.laplacian(), p.find_edges(), p.cast(), p.rescale(),
    p.normalize(scale=255), p.clamp(), p.channels_first(), p.channels_last(),
    p.add_batch_dimension(), p.remove_batch_dimension(), p.random_apply(p.invert(), 1),
    p.random_choice([p.invert()]), p.random_crop((20, 20)),
    p.random_resize((20, 20), (30, 30)), p.random_flip(1),
    p.random_rotate((0, 0)), p.color_jitter(0.1, 0.1, 0.1, 0.1),
    p.random_erasing(1), p.seed(7),
])
def test_each_single_image_operation_smoke(tmp_path, operation):
    image = media.load(_image_path(tmp_path))[0]
    result = p.materialize(operation(image), output="native", cache=False)
    assert result is not None


def test_terminal_and_multi_image_operations(tmp_path):
    image = media.load(_image_path(tmp_path))[0]
    other = Image.new("RGB", (100, 50), "blue")
    mask = Image.new("L", (100, 50), 128)

    assert len(p.materialize(p.tile((30, 20))(image), output="native")) > 1
    assert p.materialize(p.tensor()(image), output="native").ndim == 3

    for operation in (
        p.apply_mask(mask), p.composite(other, Image.new("RGB", other.size), mask),
        p.blend(other), p.add(other), p.subtract(other), p.difference(other),
        p.multiply(other), p.alpha_composite(other),
    ):
        assert p.materialize(operation(image), output="native") is not None


def test_statistics_and_hash_metrics(tmp_path):
    image = media.load(_image_path(tmp_path))[0]
    assert p.histogram(image).shape == (256,)
    assert isinstance(p.mean(image), float)
    assert isinstance(p.std(image), float)
    assert len(p.min_max(image)) == 2
    assert isinstance(p.percentile(50, image), float)
    assert isinstance(p.perceptual_hash(image), str)
    assert p.similarity_score(image)(image) == 1.0
