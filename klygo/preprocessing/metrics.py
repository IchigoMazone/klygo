from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from .core import Metric, _to_array, materialize


def _preview(image: Any, preview_size: Optional[Tuple[int, int]] = (160, 160), grayscale: bool = False) -> np.ndarray:
    value = materialize(image, output="array", cache=False) if image.__class__.__name__ == "LazyImage" else _to_array(image)
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if preview_size is not None:
        array = cv.resize(array, tuple(preview_size), interpolation=cv.INTER_AREA)
    if grayscale and array.ndim == 3:
        if array.shape[-1] == 4:
            array = array[..., :3]
        array = cv.cvtColor(array, cv.COLOR_RGB2GRAY)
    return array


def brightness_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    return Metric("brightness_score", lambda image: float(_preview(image, preview_size, True).mean()), {"preview_size": preview_size})


def contrast_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    return Metric("contrast_score", lambda image: float(_preview(image, preview_size, True).std()), {"preview_size": preview_size})


def sharpness_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    return Metric("sharpness_score", lambda image: float(cv.Laplacian(_preview(image, preview_size, True), cv.CV_64F).var()), {"preview_size": preview_size})


def blur_score(method: str = "laplacian", preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    method = method.lower()

    def evaluate(image: Any) -> float:
        gray = _preview(image, preview_size, True)
        if method == "laplacian":
            sharpness = float(cv.Laplacian(gray, cv.CV_64F).var())
        elif method == "tenengrad":
            gx = cv.Sobel(gray, cv.CV_64F, 1, 0)
            gy = cv.Sobel(gray, cv.CV_64F, 0, 1)
            sharpness = float(np.mean(gx * gx + gy * gy))
        else:
            raise ValueError("method must be 'laplacian' or 'tenengrad'")
        return 1.0 / (1.0 + sharpness)

    return Metric("blur_score", evaluate, {"method": method, "preview_size": preview_size})


def entropy_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    def evaluate(image: Any) -> float:
        gray = _preview(image, preview_size, True)
        histogram_values = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        probabilities = histogram_values / max(1.0, histogram_values.sum())
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    return Metric("entropy_score", evaluate, {"preview_size": preview_size})


def noise_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    def evaluate(image: Any) -> float:
        gray = _preview(image, preview_size, True).astype(np.float32)
        smooth = cv.GaussianBlur(gray, (3, 3), 0)
        return float(np.std(gray - smooth))

    return Metric("noise_score", evaluate, {"preview_size": preview_size})


def exposure_score(preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    def evaluate(image: Any) -> float:
        gray = _preview(image, preview_size, True).astype(np.float32) / 255.0
        return float(1.0 - min(1.0, abs(float(gray.mean()) - 0.5) * 2.0))

    return Metric("exposure_score", evaluate, {"preview_size": preview_size})


def similarity_score(other: Any, method: str = "phash", preview_size: Tuple[int, int] = (32, 32)) -> Metric:
    method = method.lower()

    def evaluate(image: Any) -> float:
        first = _preview(image, preview_size, True)
        second = _preview(other, preview_size, True)
        if method == "phash":
            a, b = _phash(first), _phash(second)
            return float(1.0 - np.mean(a != b))
        if method == "mse":
            error = float(np.mean((first.astype(np.float32) - second.astype(np.float32)) ** 2))
            return float(1.0 / (1.0 + error))
        if method == "histogram":
            hist_a = cv.normalize(cv.calcHist([first], [0], None, [256], [0, 256]), None).flatten()
            hist_b = cv.normalize(cv.calcHist([second], [0], None, [256], [0, 256]), None).flatten()
            return float(cv.compareHist(hist_a, hist_b, cv.HISTCMP_CORREL))
        raise ValueError("method must be 'phash', 'mse', or 'histogram'")

    return Metric("similarity_score", evaluate, {"method": method, "preview_size": preview_size})


def motion_score(previous: Any, preview_size: Tuple[int, int] = (160, 160)) -> Metric:
    def evaluate(image: Any) -> float:
        first = _preview(previous, preview_size, True).astype(np.float32)
        second = _preview(image, preview_size, True).astype(np.float32)
        return float(np.mean(cv.absdiff(first, second)) / 255.0)

    return Metric("motion_score", evaluate, {"preview_size": preview_size})


def _phash(gray: np.ndarray) -> np.ndarray:
    resized = cv.resize(gray, (32, 32), interpolation=cv.INTER_AREA).astype(np.float32)
    coefficients = cv.dct(resized)[:8, :8]
    return coefficients > np.median(coefficients[1:])


def perceptual_hash(image: Any = None, method: str = "phash") -> Any:
    def evaluate(value: Any) -> str:
        gray = _preview(value, (32, 32), True)
        if method.lower() == "phash":
            bits = _phash(gray).flatten()
        elif method.lower() == "dhash":
            small = cv.resize(gray, (9, 8), interpolation=cv.INTER_AREA)
            bits = (small[:, 1:] > small[:, :-1]).flatten()
        else:
            raise ValueError("method must be 'phash' or 'dhash'")
        return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"

    metric = Metric("perceptual_hash", evaluate, {"method": method})
    return metric if image is None else metric(image)


def histogram(image: Any = None, channel: Optional[int] = None, bins: int = 256) -> Any:
    def evaluate(value: Any) -> np.ndarray:
        array = _preview(value, None, False)
        selected = array if array.ndim == 2 else array[..., channel or 0]
        return np.histogram(selected, bins=bins, range=(0, 256))[0]

    metric = Metric("histogram", evaluate, {"channel": channel, "bins": bins})
    return metric if image is None else metric(image)


def mean(image: Any = None, channel: Optional[int] = None) -> Any:
    metric = Metric("mean", lambda value: float((_preview(value, None, False) if channel is None else _preview(value, None, False)[..., channel]).mean()), {"channel": channel})
    return metric if image is None else metric(image)


def std(image: Any = None, channel: Optional[int] = None) -> Any:
    metric = Metric("std", lambda value: float((_preview(value, None, False) if channel is None else _preview(value, None, False)[..., channel]).std()), {"channel": channel})
    return metric if image is None else metric(image)


def min_max(image: Any = None, channel: Optional[int] = None) -> Any:
    def evaluate(value: Any) -> Tuple[float, float]:
        array = _preview(value, None, False)
        if channel is not None:
            array = array[..., channel]
        return float(array.min()), float(array.max())

    metric = Metric("min_max", evaluate, {"channel": channel})
    return metric if image is None else metric(image)


def percentile(value: float, image: Any = None, channel: Optional[int] = None) -> Any:
    def evaluate(target: Any) -> float:
        array = _preview(target, None, False)
        if channel is not None:
            array = array[..., channel]
        return float(np.percentile(array, value))

    metric = Metric("percentile", evaluate, {"value": value, "channel": channel})
    return metric if image is None else metric(image)


__all__ = [
    "brightness_score",
    "contrast_score",
    "sharpness_score",
    "blur_score",
    "entropy_score",
    "noise_score",
    "exposure_score",
    "motion_score",
    "similarity_score",
    "perceptual_hash",
    "histogram",
    "mean",
    "std",
    "min_max",
    "percentile",
]
