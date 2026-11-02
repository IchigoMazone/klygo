from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps


def get_trace(image: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in getattr(image, "_processing_trace", [])]


def original_size(image: Any) -> Tuple[int, int]:
    stored = getattr(image, "_processing_original_size", None)
    if stored is not None:
        return tuple(stored)
    trace = get_trace(image)
    if trace:
        return tuple(trace[0]["before"])
    return tuple(image.size)


def processed_size(image: Any) -> Tuple[int, int]:
    trace = get_trace(image)
    if trace:
        return tuple(trace[-1]["after"])
    return tuple(image.size)


def clear_trace(image: Any) -> Any:
    image._processing_trace = []
    image._processing_original_size = None
    return image


def _restore_point(point: Sequence[float], item: Dict[str, Any]) -> Tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    name = item["name"]
    before = tuple(item["before"])
    after = tuple(item["after"])

    if name in ("resize", "thumbnail", "contain", "random_resize"):
        sx, sy = item.get("scale", (after[0] / before[0], after[1] / before[1]))
        return x / sx, y / sy
    if name in ("cover", "fit"):
        crop_left, crop_top = item["crop"]
        scale = item["scale"]
        return (x + crop_left) / scale, (y + crop_top) / scale
    if name in ("crop", "center_crop", "random_crop"):
        left, top, _, _ = item["box"]
        return x + left, y + top
    if name in ("letterbox",):
        left, top, _, _ = item["padding"]
        scale = item["scale"]
        return (x - left) / scale, (y - top) / scale
    if name == "pad":
        left, top, _, _ = item["padding"]
        return x - left, y - top
    if name == "flip":
        if item.get("horizontal"):
            x = before[0] - x
        if item.get("vertical"):
            y = before[1] - y
        return x, y
    if name == "rotate":
        angle = -float(item["angle"])
        source_center = np.array([before[0] / 2.0, before[1] / 2.0])
        output_center = np.array([after[0] / 2.0, after[1] / 2.0])
        radians = np.deg2rad(angle)
        matrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        restored = matrix @ (np.array([x, y]) - output_center) + source_center
        return float(restored[0]), float(restored[1])
    if name == "transpose":
        method = str(item.get("method", "")).lower()
        if method == "flip_left_right":
            return before[0] - x, y
        if method == "flip_top_bottom":
            return x, before[1] - y
        if method == "rotate_90":
            return before[0] - y, x
        if method == "rotate_180":
            return before[0] - x, before[1] - y
        if method == "rotate_270":
            return y, before[1] - x
        if method == "transpose":
            return y, x
        if method == "transverse":
            return before[0] - y, before[1] - x
    if name == "auto_orient":
        orientation = item.get("orientation", 1)
        mappings = {
            2: lambda px, py: (before[0] - px, py),
            3: lambda px, py: (before[0] - px, before[1] - py),
            4: lambda px, py: (px, before[1] - py),
            5: lambda px, py: (py, px),
            6: lambda px, py: (py, before[1] - px),
            7: lambda px, py: (before[0] - py, before[1] - px),
            8: lambda px, py: (before[0] - py, px),
        }
        return mappings.get(orientation, lambda px, py: (px, py))(x, y)
    if name in ("affine", "perspective", "warp"):
        matrix = item["matrix"]
        if len(matrix) == 6:
            a, b, c, d, e, f = matrix
            return a * x + b * y + c, d * x + e * y + f
        a, b, c, d, e, f, g, h = matrix
        denominator = g * x + h * y + 1.0
        return (a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator
    return x, y


def restore_points(points: Iterable[Sequence[float]], trace: Iterable[Dict[str, Any]]) -> List[List[float]]:
    restored = [tuple(map(float, point[:2])) for point in points]
    for item in reversed(list(trace)):
        restored = [_restore_point(point, item) for point in restored]
    return [[float(x), float(y)] for x, y in restored]


def restore_boxes(boxes: Iterable[Sequence[float]], trace: Iterable[Dict[str, Any]]) -> List[List[float]]:
    output: List[List[float]] = []
    active_trace = list(trace)
    for box in boxes:
        x1, y1, x2, y2 = map(float, box[:4])
        corners = restore_points(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), active_trace)
        xs, ys = [point[0] for point in corners], [point[1] for point in corners]
        output.append([min(xs), min(ys), max(xs), max(ys)])
    return output


def restore_mask(mask: Any, trace: Iterable[Dict[str, Any]]) -> Image.Image:
    result = mask if isinstance(mask, Image.Image) else Image.fromarray(np.asarray(mask))
    for item in reversed(list(trace)):
        name = item["name"]
        before = tuple(item["before"])
        if name in ("resize", "thumbnail", "contain", "random_resize", "letterbox"):
            if name == "letterbox":
                left, top, right, bottom = item["padding"]
                result = result.crop((left, top, result.width - right, result.height - bottom))
            result = result.resize(before, Image.Resampling.NEAREST)
        elif name in ("cover", "fit"):
            scale = item["scale"]
            resized_size = (max(1, round(before[0] * scale)), max(1, round(before[1] * scale)))
            left, top = item["crop"]
            canvas = Image.new(result.mode, resized_size, 0)
            canvas.paste(result, (left, top))
            result = canvas.resize(before, Image.Resampling.NEAREST)
        elif name in ("crop", "center_crop", "random_crop"):
            left, top, right, bottom = item["box"]
            canvas = Image.new(result.mode, before, 0)
            canvas.paste(result.resize((right - left, bottom - top), Image.Resampling.NEAREST), (left, top))
            result = canvas
        elif name == "pad":
            left, top, right, bottom = item["padding"]
            result = result.crop((left, top, result.width - right, result.height - bottom))
        elif name == "flip":
            if item.get("vertical"):
                result = ImageOps.flip(result)
            if item.get("horizontal"):
                result = ImageOps.mirror(result)
        elif name == "rotate":
            result = result.rotate(-item["angle"], Image.Resampling.NEAREST, expand=item.get("expand", False)).resize(before, Image.Resampling.NEAREST)
        elif name == "transpose":
            inverse_methods = {
                "flip_left_right": Image.Transpose.FLIP_LEFT_RIGHT,
                "flip_top_bottom": Image.Transpose.FLIP_TOP_BOTTOM,
                "rotate_90": Image.Transpose.ROTATE_270,
                "rotate_180": Image.Transpose.ROTATE_180,
                "rotate_270": Image.Transpose.ROTATE_90,
                "transpose": Image.Transpose.TRANSPOSE,
                "transverse": Image.Transpose.TRANSVERSE,
            }
            result = result.transpose(inverse_methods[str(item.get("method", "")).lower()])
        elif name == "auto_orient":
            inverse_orientations = {
                2: Image.Transpose.FLIP_LEFT_RIGHT,
                3: Image.Transpose.ROTATE_180,
                4: Image.Transpose.FLIP_TOP_BOTTOM,
                5: Image.Transpose.TRANSPOSE,
                6: Image.Transpose.ROTATE_90,
                7: Image.Transpose.TRANSVERSE,
                8: Image.Transpose.ROTATE_270,
            }
            orientation = item.get("orientation", 1)
            if orientation in inverse_orientations:
                result = result.transpose(inverse_orientations[orientation])
    return result


__all__ = [
    "get_trace",
    "original_size",
    "processed_size",
    "restore_boxes",
    "restore_points",
    "restore_mask",
    "clear_trace",
]
