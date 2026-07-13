from pathlib import Path
from typing import Any, Iterator, List, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image

from klygo.io.read_images import read_images


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _is_video(source: Any) -> bool:
    return (
        isinstance(source, (str, Path))
        and Path(source).is_file()
        and Path(source).suffix.lower() in VIDEO_SUFFIXES
    )


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGRA2RGB))
        raise ValueError(f"Unsupported OpenCV image shape: {image.shape}")
    raise TypeError(
        "source images must be PIL.Image.Image or OpenCV NumPy arrays"
    )


def _normalize_images(source: Any) -> List[Image.Image]:
    if isinstance(source, (str, Path)):
        if _is_video(source):
            raise ValueError("Video source is not supported by predict")
        return [
            _to_pil_image(image)
            for image in read_images(source, backend="pil")
        ]

    values = list(source) if isinstance(source, (list, tuple)) else [source]
    if not values:
        raise ValueError("source must contain at least one image")
    return [_to_pil_image(image) for image in values]


def _iter_media(
    source: Any,
) -> Tuple[
    Iterator[Tuple[Image.Image, np.ndarray]],
    int,
    Union[int, None],
    Union[Tuple[int, int], None],
    bool,
]:
    if _is_video(source):
        cap = cv.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {source}")
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        size = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )

        def _video_frames() -> Iterator[Tuple[Image.Image, np.ndarray]]:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    yield image, frame
            finally:
                cap.release()

        return _video_frames(), total, fps, size, True

    images = _normalize_images(source)

    def _image_frames() -> Iterator[Tuple[Image.Image, np.ndarray]]:
        for image in images:
            frame = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
            yield image, frame

    return _image_frames(), len(images), None, None, False
