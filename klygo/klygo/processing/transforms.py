from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .core import Compose, Operation, _to_array, _to_pil, compose, execute_operation


Size = Union[int, Tuple[int, int]]


def _pair(value: Size) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("size must be an int or (width, height)")
    return int(value[0]), int(value[1])


def _operation(name: str, backend: str = "auto", geometry: bool = False, terminal: bool = False, **params: Any) -> Operation:
    return Operation(name, params, backend, True, geometry, terminal)


# Geometry
def crop(box: Sequence[int]) -> Operation:
    return _operation("crop", "pil", True, box=tuple(int(v) for v in box))


def center_crop(size: Size) -> Operation:
    return _operation("center_crop", "pil", True, size=_pair(size))


def resize(size: Size, keep_ratio: bool = False, interpolation: str = "bilinear") -> Operation:
    return _operation("resize", "pil", True, size=_pair(size), keep_ratio=keep_ratio, interpolation=interpolation)


def thumbnail(size: Size, interpolation: str = "bilinear") -> Operation:
    return _operation("thumbnail", "pil", True, size=_pair(size), interpolation=interpolation)


def contain(size: Size, interpolation: str = "bilinear") -> Operation:
    return _operation("contain", "pil", True, size=_pair(size), interpolation=interpolation)


def cover(size: Size, interpolation: str = "bilinear") -> Operation:
    return _operation("cover", "pil", True, size=_pair(size), interpolation=interpolation)


def fit(size: Size, interpolation: str = "bilinear", centering: Tuple[float, float] = (0.5, 0.5)) -> Operation:
    return _operation("fit", "pil", True, size=_pair(size), interpolation=interpolation, centering=centering)


def letterbox(size: Size, color: Any = (114, 114, 114), stride: Optional[int] = None) -> Operation:
    return _operation("letterbox", "pil", True, size=_pair(size), color=color, stride=stride)


def pad(padding: Union[int, Sequence[int]], color: Any = 0) -> Operation:
    normalized = padding if isinstance(padding, int) else tuple(int(v) for v in padding)
    return _operation("pad", "pil", True, padding=normalized, color=color)


def flip(horizontal: bool = False, vertical: bool = False) -> Operation:
    return _operation("flip", "pil", True, horizontal=horizontal, vertical=vertical)


def rotate(angle: float, expand: bool = False, fill: Any = 0, interpolation: str = "bilinear") -> Operation:
    return _operation("rotate", "pil", True, angle=angle, expand=expand, fill=fill, interpolation=interpolation)


def transpose(method: str) -> Operation:
    return _operation("transpose", "pil", True, method=method)


def affine(matrix: Sequence[float], size: Optional[Size] = None, fill: Any = 0) -> Operation:
    return _operation("affine", "pil", True, matrix=tuple(matrix), size=_pair(size) if size is not None else None, fill=fill)


def perspective(matrix: Sequence[float], size: Optional[Size] = None, fill: Any = 0) -> Operation:
    return _operation("perspective", "pil", True, matrix=tuple(matrix), size=_pair(size) if size is not None else None, fill=fill)


def warp(matrix: Sequence[float], size: Optional[Size] = None, fill: Any = 0) -> Operation:
    return _operation("warp", "pil", True, matrix=tuple(matrix), size=_pair(size) if size is not None else None, fill=fill)


def tile(size: Size, overlap: float = 0.0) -> Operation:
    return _operation("tile", "pil", True, True, size=_pair(size), overlap=float(overlap))


# Orientation / EXIF
def auto_orient() -> Operation:
    return _operation("auto_orient", "pil", True)


def exif_transpose() -> Operation:
    return auto_orient()


def strip_exif() -> Operation:
    return _operation("strip_exif", "pil")


# Color and channels
def convert_color(source: str, target: Optional[str] = None) -> Operation:
    if target is None:
        target = source
        source = "AUTO"
    return _operation("convert_color", "opencv", source=source.upper(), target=target.upper())


def grayscale(channels: int = 1) -> Operation:
    return _operation("grayscale", "pil", channels=int(channels))


def rgb() -> Operation:
    return convert_color("RGB")


def bgr() -> Operation:
    return convert_color("BGR")


def rgba() -> Operation:
    return convert_color("RGBA")


def add_alpha(value: int = 255) -> Operation:
    return _operation("add_alpha", "numpy", value=int(value))


def remove_alpha(background: Any = (255, 255, 255)) -> Operation:
    return _operation("remove_alpha", "pil", background=background)


def select_channels(channels: Sequence[int]) -> Operation:
    return _operation("select_channels", "numpy", channels=tuple(channels))


def swap_channels(order: Sequence[int]) -> Operation:
    return _operation("swap_channels", "numpy", order=tuple(order))


# Tone and enhancement
def brightness(factor: float) -> Operation:
    return _operation("brightness", "pil", factor=float(factor))


def contrast(factor: float) -> Operation:
    return _operation("contrast", "pil", factor=float(factor))


def saturation(factor: float) -> Operation:
    return _operation("saturation", "pil", factor=float(factor))


def hue(value: float) -> Operation:
    return _operation("hue", "opencv", value=float(value))


def gamma(value: float) -> Operation:
    return _operation("gamma", "numpy", value=float(value))


def temperature(value: float) -> Operation:
    return _operation("temperature", "numpy", value=float(value))


def tint(value: float) -> Operation:
    return _operation("tint", "numpy", value=float(value))


def white_balance(method: str = "gray_world") -> Operation:
    return _operation("white_balance", "opencv", method=method)


def autocontrast(cutoff: Union[int, float, Tuple[float, float]] = 0) -> Operation:
    return _operation("autocontrast", "pil", cutoff=cutoff)


def equalize() -> Operation:
    return _operation("equalize", "pil")


def clahe(clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> Operation:
    return _operation("clahe", "opencv", clip_limit=float(clip_limit), grid_size=tuple(grid_size))


def invert() -> Operation:
    return _operation("invert", "pil")


def solarize(threshold: int = 128) -> Operation:
    return _operation("solarize", "pil", threshold=int(threshold))


def posterize(bits: int) -> Operation:
    return _operation("posterize", "pil", bits=int(bits))


def colorize(black: Any, white: Any, mid: Any = None) -> Operation:
    return _operation("colorize", "pil", black=black, white=white, mid=mid)


# Filters
def blur(radius: float = 1) -> Operation:
    return box_blur(radius)


def box_blur(radius: float = 1) -> Operation:
    return _operation("box_blur", "pil", radius=float(radius))


def gaussian_blur(kernel: int = 3, sigma: float = 0) -> Operation:
    return _operation("gaussian_blur", "opencv", kernel=int(kernel), sigma=float(sigma))


def median_blur(kernel: int = 3) -> Operation:
    return _operation("median_blur", "opencv", kernel=int(kernel))


def bilateral_filter(diameter: int = 5, sigma_color: float = 50, sigma_space: float = 50) -> Operation:
    return _operation("bilateral_filter", "opencv", diameter=int(diameter), sigma_color=float(sigma_color), sigma_space=float(sigma_space))


def sharpen(amount: float = 1.0) -> Operation:
    return _operation("sharpen", "pil", amount=float(amount))


def unsharp_mask(radius: float = 2, amount: float = 1.0, threshold: int = 0) -> Operation:
    return _operation("unsharp_mask", "pil", radius=float(radius), amount=float(amount), threshold=int(threshold))


def denoise(strength: float = 5) -> Operation:
    return _operation("denoise", "opencv", strength=float(strength))


def detail() -> Operation:
    return _operation("detail", "pil")


def smooth() -> Operation:
    return _operation("smooth", "pil")


def emboss() -> Operation:
    return _operation("emboss", "pil")


def convolve(kernel: Sequence[Sequence[float]], scale: Optional[float] = None, offset: float = 0) -> Operation:
    return _operation("convolve", "opencv", kernel=np.asarray(kernel, dtype=np.float32), scale=scale, offset=float(offset))


# Threshold
def threshold(value: float, mode: str = "binary", max_value: float = 255) -> Operation:
    return _operation("threshold", "opencv", value=float(value), mode=mode, max_value=float(max_value))


def adaptive_threshold(method: str = "gaussian", block_size: int = 11, constant: float = 2, mode: str = "binary") -> Operation:
    return _operation("adaptive_threshold", "opencv", method=method, block_size=int(block_size), constant=float(constant), mode=mode)


def otsu_threshold(mode: str = "binary") -> Operation:
    return _operation("otsu_threshold", "opencv", mode=mode)


def triangle_threshold(mode: str = "binary") -> Operation:
    return _operation("triangle_threshold", "opencv", mode=mode)


# Morphology
def morphology(operation: str, kernel: Union[int, Sequence[Sequence[int]]] = 3, iterations: int = 1) -> Operation:
    return _operation("morphology", "opencv", operation=operation, kernel=kernel, iterations=int(iterations))


def erode(kernel: Any = 3, iterations: int = 1) -> Operation:
    return morphology("erode", kernel, iterations)


def dilate(kernel: Any = 3, iterations: int = 1) -> Operation:
    return morphology("dilate", kernel, iterations)


def opening(kernel: Any = 3, iterations: int = 1) -> Operation:
    return morphology("open", kernel, iterations)


def closing(kernel: Any = 3, iterations: int = 1) -> Operation:
    return morphology("close", kernel, iterations)


def morphological_gradient(kernel: Any = 3) -> Operation:
    return morphology("gradient", kernel)


def tophat(kernel: Any = 3) -> Operation:
    return morphology("tophat", kernel)


def blackhat(kernel: Any = 3) -> Operation:
    return morphology("blackhat", kernel)


# Edges
def canny(low: float = 100, high: float = 200, aperture_size: int = 3) -> Operation:
    return _operation("canny", "opencv", low=float(low), high=float(high), aperture_size=int(aperture_size))


def sobel(dx: int = 1, dy: int = 0, kernel: int = 3) -> Operation:
    return _operation("sobel", "opencv", dx=int(dx), dy=int(dy), kernel=int(kernel))


def scharr(dx: int = 1, dy: int = 0) -> Operation:
    return _operation("scharr", "opencv", dx=int(dx), dy=int(dy))


def laplacian(kernel: int = 3) -> Operation:
    return _operation("laplacian", "opencv", kernel=int(kernel))


def find_edges() -> Operation:
    return _operation("find_edges", "pil")


# Model input
def cast(dtype: str = "float32") -> Operation:
    return _operation("cast", "numpy", dtype=dtype)


def rescale(scale: float = 1 / 255) -> Operation:
    return _operation("rescale", "numpy", scale=float(scale))


def normalize(mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None, scale: Optional[float] = None) -> Operation:
    return _operation("normalize", "numpy", mean=tuple(mean) if mean is not None else None, std=tuple(std) if std is not None else None, scale=scale)


def clamp(min_value: float = 0, max_value: float = 1) -> Operation:
    return _operation("clamp", "numpy", min_value=min_value, max_value=max_value)


def channels_first() -> Operation:
    return _operation("channels_first", "numpy")


def channels_last() -> Operation:
    return _operation("channels_last", "numpy")


def add_batch_dimension() -> Operation:
    return _operation("add_batch_dimension", "numpy")


def remove_batch_dimension() -> Operation:
    return _operation("remove_batch_dimension", "numpy")


def tensor(dtype: str = "float32", device: Optional[str] = None) -> Operation:
    return _operation("tensor", "torch", dtype=dtype, device=device)


def from_input_spec(input_spec: Any) -> Compose:
    getter = input_spec.get if isinstance(input_spec, dict) else lambda key, default=None: getattr(input_spec, key, default)
    operations: List[Operation] = []
    size = getter("size")
    if size:
        operations.append(letterbox(size) if getter("letterbox", False) else resize(size))
    color = getter("color", getter("color_space"))
    if color:
        operations.append(convert_color(color))
    mean, std = getter("mean"), getter("std")
    scale = getter("scale")
    if mean is not None or std is not None or scale is not None:
        operations.append(normalize(mean, std, scale))
    layout = str(getter("layout", "")).upper()
    if layout in ("CHW", "NCHW"):
        operations.append(channels_first())
    dtype = getter("dtype")
    if dtype:
        operations.append(cast(str(dtype)))
    return compose(operations)


# Multiple images
def apply_mask(mask: Any) -> Operation:
    return _operation("apply_mask", "pil", mask=mask)


def composite(foreground: Any, background: Any, mask: Any) -> Operation:
    return _operation("composite", "pil", True, foreground=foreground, background=background, mask=mask)


def blend(other: Any, alpha: float = 0.5) -> Operation:
    return _operation("blend", "pil", other=other, alpha=float(alpha))


def add(other: Any, scale: float = 1.0) -> Operation:
    return _operation("add", "opencv", other=other, scale=float(scale))


def subtract(other: Any) -> Operation:
    return _operation("subtract", "opencv", other=other)


def difference(other: Any) -> Operation:
    return _operation("difference", "opencv", other=other)


def multiply(other: Any) -> Operation:
    return _operation("multiply", "opencv", other=other)


def alpha_composite(other: Any) -> Operation:
    return _operation("alpha_composite", "pil", other=other)


# Augmentation
def random_apply(operation: Union[Operation, Compose], probability: float = 0.5) -> Operation:
    return _operation("random_apply", "auto", operation=operation, probability=float(probability))


def random_choice(operations: Sequence[Union[Operation, Compose]]) -> Operation:
    return _operation("random_choice", "auto", operations=tuple(operations))


def random_crop(size: Size) -> Operation:
    return _operation("random_crop", "pil", True, size=_pair(size))


def random_resize(min_size: Size, max_size: Size) -> Operation:
    return _operation("random_resize", "pil", True, min_size=_pair(min_size), max_size=_pair(max_size))


def random_flip(probability: float = 0.5, horizontal: bool = True, vertical: bool = False) -> Operation:
    return _operation("random_flip", "pil", True, probability=float(probability), horizontal=horizontal, vertical=vertical)


def random_rotate(angle_range: Tuple[float, float] = (-10, 10), expand: bool = False) -> Operation:
    return _operation("random_rotate", "pil", True, angle_range=tuple(angle_range), expand=expand)


def color_jitter(brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0) -> Operation:
    return _operation("color_jitter", "pil", brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)


def random_erasing(probability: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), value: Any = 0) -> Operation:
    return _operation("random_erasing", "numpy", probability=float(probability), scale=tuple(scale), value=value)


def seed(value: int) -> Operation:
    return Operation("seed", {"value": int(value)}, "auto", False)


def _resampling(name: str) -> Image.Resampling:
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(str(name).lower(), Image.Resampling.BILINEAR)


def _size(value: Any) -> Tuple[int, int]:
    if isinstance(value, Image.Image):
        return value.size
    array = _to_array(value)
    if array.ndim >= 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        return int(array.shape[2]), int(array.shape[1])
    return int(array.shape[1]), int(array.shape[0])


def _trace(trace: List[Dict[str, Any]], operation: Operation, before: Tuple[int, int], after: Tuple[int, int], **extra: Any) -> None:
    if operation.changes_geometry:
        trace.append({"name": operation.name, "before": before, "after": after, **extra})


def _gray_array(value: Any) -> np.ndarray:
    array = _to_array(value)
    if array.ndim == 2:
        return array.astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return cv.cvtColor(array.astype(np.uint8), cv.COLOR_RGB2GRAY)


def _kernel(value: Any) -> np.ndarray:
    if isinstance(value, int):
        return np.ones((value, value), dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)


def _resolve_other(value: Any) -> Any:
    from .core import materialize

    if value.__class__.__name__ == "LazyImage":
        return materialize(value, "pil", cache=False)
    return value


def execute_builtin(value: Any, operation: Operation, trace: List[Dict[str, Any]]) -> Any:
    name, p = operation.name, operation.params
    before = _size(value) if operation.changes_geometry and name != "tile" else None

    if name == "pil_method":
        return getattr(_to_pil(value), p["method"])(*p["args"], **p["kwargs"])

    if name == "crop":
        result = _to_pil(value).crop(p["box"])
        _trace(trace, operation, before, result.size, box=p["box"])
        return result
    if name == "center_crop":
        image, (width, height) = _to_pil(value), p["size"]
        left, top = max(0, (image.width - width) // 2), max(0, (image.height - height) // 2)
        result = image.crop((left, top, left + width, top + height))
        _trace(trace, operation, before, result.size, box=(left, top, left + width, top + height))
        return result
    if name in ("resize", "thumbnail", "contain"):
        image, target = _to_pil(value), p["size"]
        if name != "resize" or p.get("keep_ratio"):
            result = ImageOps.contain(image, target, _resampling(p.get("interpolation", "bilinear")))
        else:
            result = image.resize(target, _resampling(p.get("interpolation", "bilinear")))
        _trace(trace, operation, before, result.size, scale=(result.width / before[0], result.height / before[1]))
        return result
    if name in ("cover", "fit"):
        image, target = _to_pil(value), p["size"]
        scale_value = max(target[0] / image.width, target[1] / image.height)
        resized_size = (max(1, round(image.width * scale_value)), max(1, round(image.height * scale_value)))
        resized = image.resize(resized_size, _resampling(p["interpolation"]))
        centering = p.get("centering", (0.5, 0.5))
        left = max(0, round((resized.width - target[0]) * centering[0]))
        top = max(0, round((resized.height - target[1]) * centering[1]))
        result = resized.crop((left, top, left + target[0], top + target[1]))
        _trace(trace, operation, before, result.size, scale=scale_value, crop=(left, top))
        return result
    if name == "letterbox":
        image, target = _to_pil(value), p["size"]
        scale_value = min(target[0] / image.width, target[1] / image.height)
        resized_size = (max(1, round(image.width * scale_value)), max(1, round(image.height * scale_value)))
        resized = image.resize(resized_size, Image.Resampling.BILINEAR)
        output_size = target
        if p.get("stride"):
            stride_value = int(p["stride"])
            output_size = tuple(int(np.ceil(v / stride_value) * stride_value) for v in target)
        left, top = (output_size[0] - resized.width) // 2, (output_size[1] - resized.height) // 2
        result = Image.new(image.mode, output_size, p["color"])
        result.paste(resized, (left, top))
        _trace(trace, operation, before, result.size, scale=scale_value, padding=(left, top, output_size[0] - resized.width - left, output_size[1] - resized.height - top))
        return result
    if name == "pad":
        result = ImageOps.expand(_to_pil(value), border=p["padding"], fill=p["color"])
        padding = p["padding"]
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        _trace(trace, operation, before, result.size, padding=tuple(padding))
        return result
    if name == "flip":
        result = _to_pil(value)
        if p["horizontal"]:
            result = ImageOps.mirror(result)
        if p["vertical"]:
            result = ImageOps.flip(result)
        _trace(trace, operation, before, result.size, horizontal=p["horizontal"], vertical=p["vertical"])
        return result
    if name == "rotate":
        result = _to_pil(value).rotate(p["angle"], _resampling(p["interpolation"]), expand=p["expand"], fillcolor=p["fill"])
        _trace(trace, operation, before, result.size, angle=p["angle"], expand=p["expand"])
        return result
    if name == "transpose":
        methods = {
            "flip_left_right": Image.Transpose.FLIP_LEFT_RIGHT,
            "flip_top_bottom": Image.Transpose.FLIP_TOP_BOTTOM,
            "rotate_90": Image.Transpose.ROTATE_90,
            "rotate_180": Image.Transpose.ROTATE_180,
            "rotate_270": Image.Transpose.ROTATE_270,
            "transpose": Image.Transpose.TRANSPOSE,
            "transverse": Image.Transpose.TRANSVERSE,
        }
        result = _to_pil(value).transpose(methods[str(p["method"]).lower()])
        _trace(trace, operation, before, result.size, method=p["method"])
        return result
    if name in ("affine", "perspective", "warp"):
        image = _to_pil(value)
        target = p.get("size") or image.size
        method = Image.Transform.AFFINE if name == "affine" else Image.Transform.PERSPECTIVE
        result = image.transform(target, method, p["matrix"], Image.Resampling.BILINEAR, fillcolor=p["fill"])
        _trace(trace, operation, before, result.size, matrix=p["matrix"])
        return result
    if name == "tile":
        image, (tile_width, tile_height) = _to_pil(value), p["size"]
        overlap = min(max(p["overlap"], 0.0), 0.95)
        step_x, step_y = max(1, int(tile_width * (1 - overlap))), max(1, int(tile_height * (1 - overlap)))
        return [image.crop((x, y, min(x + tile_width, image.width), min(y + tile_height, image.height))) for y in range(0, image.height, step_y) for x in range(0, image.width, step_x)]

    if name in ("auto_orient", "strip_exif"):
        source_image = _to_pil(value)
        orientation = source_image.getexif().get(274, 1) if name == "auto_orient" else 1
        image = ImageOps.exif_transpose(source_image) if name == "auto_orient" else source_image.copy()
        if name == "strip_exif":
            image.info.pop("exif", None)
        else:
            _trace(trace, operation, before, image.size, orientation=orientation)
        return image

    if name == "convert_color":
        source, target = p["source"], p["target"]
        image = _to_pil(value)
        if target in ("RGB", "RGBA", "L", "CMYK"):
            return image.convert(target)
        array = np.asarray(image.convert("RGB"))
        source = "RGB" if source == "AUTO" else source
        code = getattr(cv, f"COLOR_{source}2{target}", None)
        if code is None:
            raise ValueError(f"Unsupported color conversion: {source} -> {target}")
        return cv.cvtColor(array, code)
    if name == "grayscale":
        gray = ImageOps.grayscale(_to_pil(value))
        return gray if p["channels"] == 1 else gray.convert("RGB")
    if name == "add_alpha":
        array = _to_array(value)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 4:
            return array
        alpha = np.full(array.shape[:2] + (1,), p["value"], dtype=array.dtype)
        return np.concatenate((array, alpha), axis=-1)
    if name == "remove_alpha":
        image = _to_pil(value).convert("RGBA")
        background = Image.new("RGBA", image.size, tuple(p["background"]) + (255,) if len(p["background"]) == 3 else p["background"])
        return Image.alpha_composite(background, image).convert("RGB")
    if name in ("select_channels", "swap_channels"):
        key = "channels" if name == "select_channels" else "order"
        return _to_array(value)[..., list(p[key])]

    if name in ("brightness", "contrast", "saturation"):
        enhancer = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast, "saturation": ImageEnhance.Color}[name]
        return enhancer(_to_pil(value)).enhance(p["factor"])
    if name == "hue":
        array = cv.cvtColor(np.asarray(_to_pil(value).convert("RGB")), cv.COLOR_RGB2HSV)
        array[..., 0] = (array[..., 0].astype(np.int16) + int(p["value"] * 180)) % 180
        return cv.cvtColor(array, cv.COLOR_HSV2RGB)
    if name == "gamma":
        array = _to_array(value).astype(np.float32) / 255.0
        return np.clip(np.power(array, p["value"]) * 255.0, 0, 255).astype(np.uint8)
    if name in ("temperature", "tint"):
        array = _to_array(_to_pil(value).convert("RGB")).astype(np.float32)
        amount = p["value"]
        if name == "temperature":
            array[..., 0] *= 1 + amount
            array[..., 2] *= 1 - amount
        else:
            array[..., 1] *= 1 + amount
        return np.clip(array, 0, 255).astype(np.uint8)
    if name == "white_balance":
        array = _to_array(_to_pil(value).convert("RGB")).astype(np.float32)
        means = array.reshape(-1, 3).mean(axis=0)
        target = means.mean()
        return np.clip(array * (target / np.maximum(means, 1e-6)), 0, 255).astype(np.uint8)
    if name == "autocontrast":
        return ImageOps.autocontrast(_to_pil(value), cutoff=p["cutoff"])
    if name == "equalize":
        return ImageOps.equalize(_to_pil(value))
    if name == "clahe":
        image = _to_pil(value)
        array = np.asarray(image.convert("RGB"))
        lab = cv.cvtColor(array, cv.COLOR_RGB2LAB)
        lab[..., 0] = cv.createCLAHE(p["clip_limit"], p["grid_size"]).apply(lab[..., 0])
        return cv.cvtColor(lab, cv.COLOR_LAB2RGB)
    if name == "invert":
        return ImageOps.invert(_to_pil(value).convert("RGB"))
    if name == "solarize":
        return ImageOps.solarize(_to_pil(value), p["threshold"])
    if name == "posterize":
        return ImageOps.posterize(_to_pil(value).convert("RGB"), p["bits"])
    if name == "colorize":
        return ImageOps.colorize(_to_pil(value).convert("L"), p["black"], p["white"], p["mid"])

    if name == "box_blur":
        return _to_pil(value).filter(ImageFilter.BoxBlur(p["radius"]))
    if name == "gaussian_blur":
        kernel = p["kernel"] + (p["kernel"] % 2 == 0)
        return cv.GaussianBlur(_to_array(value), (kernel, kernel), p["sigma"])
    if name == "median_blur":
        kernel = max(3, p["kernel"] + (p["kernel"] % 2 == 0))
        return cv.medianBlur(_to_array(value).astype(np.uint8), kernel)
    if name == "bilateral_filter":
        return cv.bilateralFilter(_to_array(value).astype(np.uint8), p["diameter"], p["sigma_color"], p["sigma_space"])
    if name == "sharpen":
        return ImageEnhance.Sharpness(_to_pil(value)).enhance(1 + p["amount"])
    if name == "unsharp_mask":
        return _to_pil(value).filter(ImageFilter.UnsharpMask(p["radius"], int(p["amount"] * 100), p["threshold"]))
    if name == "denoise":
        array = _to_array(_to_pil(value).convert("RGB"))
        return cv.fastNlMeansDenoisingColored(array, None, p["strength"], p["strength"], 7, 21)
    if name in ("detail", "smooth", "emboss", "find_edges"):
        image_filter = {"detail": ImageFilter.DETAIL, "smooth": ImageFilter.SMOOTH, "emboss": ImageFilter.EMBOSS, "find_edges": ImageFilter.FIND_EDGES}[name]
        return _to_pil(value).filter(image_filter)
    if name == "convolve":
        kernel = p["kernel"]
        result = cv.filter2D(_to_array(value), -1, kernel)
        if p["scale"] is not None:
            result = result / p["scale"]
        return np.clip(result + p["offset"], 0, 255).astype(np.uint8)

    threshold_modes = {"binary": cv.THRESH_BINARY, "binary_inv": cv.THRESH_BINARY_INV, "truncate": cv.THRESH_TRUNC, "to_zero": cv.THRESH_TOZERO, "to_zero_inv": cv.THRESH_TOZERO_INV}
    if name == "threshold":
        _, result = cv.threshold(_gray_array(value), p["value"], p["max_value"], threshold_modes[p["mode"]])
        return result
    if name == "adaptive_threshold":
        method = cv.ADAPTIVE_THRESH_GAUSSIAN_C if p["method"].lower() == "gaussian" else cv.ADAPTIVE_THRESH_MEAN_C
        block_size = max(3, p["block_size"] + (p["block_size"] % 2 == 0))
        return cv.adaptiveThreshold(_gray_array(value), 255, method, threshold_modes[p["mode"]], block_size, p["constant"])
    if name in ("otsu_threshold", "triangle_threshold"):
        extra = cv.THRESH_OTSU if name == "otsu_threshold" else cv.THRESH_TRIANGLE
        _, result = cv.threshold(_gray_array(value), 0, 255, threshold_modes[p["mode"]] | extra)
        return result

    if name == "morphology":
        array, kernel = _to_array(value), _kernel(p["kernel"])
        if p["operation"] == "erode":
            return cv.erode(array, kernel, iterations=p["iterations"])
        if p["operation"] == "dilate":
            return cv.dilate(array, kernel, iterations=p["iterations"])
        codes = {"open": cv.MORPH_OPEN, "close": cv.MORPH_CLOSE, "gradient": cv.MORPH_GRADIENT, "tophat": cv.MORPH_TOPHAT, "blackhat": cv.MORPH_BLACKHAT}
        return cv.morphologyEx(array, codes[p["operation"]], kernel, iterations=p["iterations"])

    if name == "canny":
        return cv.Canny(_gray_array(value), p["low"], p["high"], apertureSize=p["aperture_size"])
    if name in ("sobel", "scharr"):
        gray = _gray_array(value)
        gradient = cv.Sobel(gray, cv.CV_32F, p["dx"], p["dy"], ksize=p["kernel"]) if name == "sobel" else cv.Scharr(gray, cv.CV_32F, p["dx"], p["dy"])
        return cv.convertScaleAbs(gradient)
    if name == "laplacian":
        return cv.convertScaleAbs(cv.Laplacian(_gray_array(value), cv.CV_32F, ksize=p["kernel"]))

    if name in ("cast", "rescale", "normalize", "clamp", "channels_first", "channels_last", "add_batch_dimension", "remove_batch_dimension"):
        array = _to_array(value)
        if name == "cast":
            return array.astype(p["dtype"])
        if name == "rescale":
            return array.astype(np.float32) * p["scale"]
        if name == "normalize":
            result = array.astype(np.float32)
            if p["scale"] is not None:
                result = result / float(p["scale"])
            mean, std = p["mean"], p["std"]
            if mean is not None:
                result = result - np.asarray(mean, dtype=np.float32)
            if std is not None:
                result = result / np.asarray(std, dtype=np.float32)
            return result
        if name == "clamp":
            return np.clip(array, p["min_value"], p["max_value"])
        if name == "channels_first" and array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return np.transpose(array, (2, 0, 1))
        if name == "channels_last" and array.ndim == 3 and array.shape[0] in (1, 3, 4):
            return np.transpose(array, (1, 2, 0))
        if name == "add_batch_dimension":
            return np.expand_dims(array, 0)
        if name == "remove_batch_dimension" and array.ndim > 0 and array.shape[0] == 1:
            return array[0]
        return array
    if name == "tensor":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for processing.tensor()") from exc
        result = torch.from_numpy(np.array(_to_array(value), copy=True, order="C"))
        dtype = getattr(torch, p["dtype"], None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {p['dtype']}")
        result = result.to(dtype=dtype)
        return result.to(p["device"]) if p["device"] else result

    if name == "apply_mask":
        image = _to_pil(value)
        mask_image = _to_pil(_resolve_other(p["mask"])).convert("L").resize(image.size)
        return Image.composite(image, Image.new(image.mode, image.size), mask_image)
    if name == "composite":
        foreground, background = _to_pil(_resolve_other(p["foreground"])), _to_pil(_resolve_other(p["background"]))
        mask_image = _to_pil(_resolve_other(p["mask"])).convert("L")
        return Image.composite(foreground, background.resize(foreground.size), mask_image.resize(foreground.size))
    if name in ("blend", "alpha_composite"):
        first, second = _to_pil(value), _to_pil(_resolve_other(p["other"])).resize(_to_pil(value).size)
        return Image.blend(first.convert("RGBA"), second.convert("RGBA"), p["alpha"]) if name == "blend" else Image.alpha_composite(first.convert("RGBA"), second.convert("RGBA"))
    if name in ("add", "subtract", "difference", "multiply"):
        first, second = _to_array(value), _to_array(_resolve_other(p["other"]))
        if first.shape != second.shape:
            second = cv.resize(second, (first.shape[1], first.shape[0]))
        if name == "add":
            return cv.addWeighted(first, p["scale"], second, p["scale"], 0)
        if name == "subtract":
            return cv.subtract(first, second)
        if name == "difference":
            return cv.absdiff(first, second)
        return cv.multiply(first, second, scale=1 / 255)

    if name == "seed":
        random.seed(p["value"])
        np.random.seed(p["value"])
        return value
    if name in ("random_apply", "random_choice"):
        selected = p["operation"] if name == "random_apply" and random.random() < p["probability"] else None
        if name == "random_choice":
            selected = random.choice(p["operations"])
        if selected is None:
            return value
        operations = selected.operations if isinstance(selected, Compose) else (selected,)
        result = value
        for selected_operation in operations:
            result = execute_operation(result, selected_operation, trace)
        return result
    if name == "random_crop":
        image, target = _to_pil(value), p["size"]
        left = random.randint(0, max(0, image.width - target[0]))
        top = random.randint(0, max(0, image.height - target[1]))
        result = image.crop((left, top, left + target[0], top + target[1]))
        _trace(trace, operation, before, result.size, box=(left, top, left + target[0], top + target[1]))
        return result
    if name == "random_resize":
        width = random.randint(p["min_size"][0], p["max_size"][0])
        height = random.randint(p["min_size"][1], p["max_size"][1])
        result = _to_pil(value).resize((width, height), Image.Resampling.BILINEAR)
        _trace(trace, operation, before, result.size, scale=(width / before[0], height / before[1]))
        return result
    if name == "random_flip":
        if random.random() >= p["probability"]:
            return value
        selected = flip(p["horizontal"], p["vertical"])
        return execute_builtin(value, selected, trace)
    if name == "random_rotate":
        selected = rotate(random.uniform(*p["angle_range"]), p["expand"])
        return execute_builtin(value, selected, trace)
    if name == "color_jitter":
        result = _to_pil(value)
        for key, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast), ("saturation", ImageEnhance.Color)):
            amount = p[key]
            if amount:
                result = enhancer(result).enhance(random.uniform(max(0, 1 - amount), 1 + amount))
        if p["hue"]:
            result = execute_builtin(result, globals()["hue"](random.uniform(-p["hue"], p["hue"])), trace)
        return result
    if name == "random_erasing":
        array = _to_array(value).copy()
        if random.random() >= p["probability"]:
            return array
        area = array.shape[0] * array.shape[1] * random.uniform(*p["scale"])
        side = max(1, int(area ** 0.5))
        height, width = min(side, array.shape[0]), min(side, array.shape[1])
        top, left = random.randint(0, array.shape[0] - height), random.randint(0, array.shape[1] - width)
        array[top : top + height, left : left + width] = p["value"]
        return array

    raise ValueError(f"Unknown processing operation: {name}")


__all__ = [
    "crop", "center_crop", "resize", "thumbnail", "contain", "cover", "fit",
    "letterbox", "pad", "flip", "rotate", "transpose", "affine", "perspective",
    "warp", "tile", "auto_orient", "exif_transpose", "strip_exif", "convert_color",
    "grayscale", "rgb", "bgr", "rgba", "add_alpha", "remove_alpha",
    "select_channels", "swap_channels", "brightness", "contrast", "saturation", "hue",
    "gamma", "temperature", "tint", "white_balance", "autocontrast", "equalize",
    "clahe", "invert", "solarize", "posterize", "colorize", "blur", "box_blur",
    "gaussian_blur", "median_blur", "bilateral_filter", "sharpen", "unsharp_mask",
    "denoise", "detail", "smooth", "emboss", "convolve", "threshold",
    "adaptive_threshold", "otsu_threshold", "triangle_threshold", "morphology", "erode",
    "dilate", "opening", "closing", "morphological_gradient", "tophat", "blackhat",
    "canny", "sobel", "scharr", "laplacian", "find_edges", "cast", "rescale",
    "normalize", "clamp", "channels_first", "channels_last", "add_batch_dimension",
    "remove_batch_dimension", "tensor", "from_input_spec", "apply_mask", "composite",
    "blend", "add", "subtract", "difference", "multiply", "alpha_composite",
    "random_apply", "random_choice", "random_crop", "random_resize", "random_flip",
    "random_rotate", "color_jitter", "random_erasing", "seed",
]
