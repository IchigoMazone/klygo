"""Backend-neutral, in-place detection postprocessing."""
import math
from typing import Callable, Iterable, Optional

import numpy as np

from klygo.outputs.detect import Box, Detection, Detections

Result = Detection | Detections


def _frames(result: Result, function: Callable[[Detection], None]) -> None:
    if isinstance(result, Detections):
        if result.is_stream:
            result.map(lambda frame: function(frame))
        else:
            with result._batch("map_frames"):
                for frame in result._frames:
                    function(frame)
    elif isinstance(result, Detection):
        function(result)
    else:
        raise TypeError("result must be a Detection or Detections")


def _select(result: Result, *, where: Optional[Callable[[Box], bool]] = None,
            labels: Optional[str | Iterable[str]] = None,
            class_ids: Optional[int | Iterable[int]] = None,
            uids: Optional[int | Iterable[int]] = None) -> None:
    """Keep boxes matching every supplied selector."""
    if where is not None and not callable(where):
        raise TypeError("predicate must be callable")
    ls = None if labels is None else ({labels} if isinstance(labels, str) else set(labels))
    cs = None if class_ids is None else ({int(class_ids)} if isinstance(class_ids, (int, np.integer)) else set(class_ids))
    us = None if uids is None else ({int(uids)} if isinstance(uids, (int, np.integer)) else set(uids))

    def select_frame(frame: Detection) -> None:
        if where is None:
            n = len(frame._labels)
            if n == 0:
                frame._slice_keep([], "filter_boxes")
                return
            cids = frame.class_ids if cs is not None else None
            uid_list = frame._uids.tolist() if us is not None else None
            keep = [
                i for i in range(n)
                if (ls is None or frame._labels[i] in ls)
                and (cs is None or cids[i] in cs)
                and (us is None or uid_list[i] in us)
            ]
            frame._slice_keep(keep, "filter_boxes")
        else:
            def matches(box: Box) -> bool:
                return (
                    bool(where(box))
                    and (ls is None or box.label in ls)
                    and (cs is None or box.class_id in cs)
                    and (us is None or box.uid in us)
                )
            frame.filter(matches)

    _frames(result, select_frame)


def _map_boxes(result: Result, function: Callable[[Box], Optional[Box]]) -> None:
    """Transform boxes atomically."""
    _frames(result, lambda frame: frame.map(function))


def _filter_score(result: Result, minimum: float) -> None:
    """Remove boxes below a score threshold."""
    minimum = float(minimum)
    if not math.isfinite(minimum):
        raise ValueError("minimum must be finite")

    def threshold_frame(frame: Detection) -> None:
        if len(frame._labels) == 0:
            frame._slice_keep([], "filter_boxes")
            return
        keep = np.flatnonzero(frame._scores >= minimum)
        frame._slice_keep(keep, "filter_boxes")

    _frames(result, threshold_frame)


def _clip_boxes(result: Result, image_size: Optional[tuple[int, int]] = None) -> None:
    """Clip coordinates and discard boxes wholly outside the image."""
    if image_size is not None:
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise TypeError("image_size must be a (width, height) pair")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in image_size):
            raise TypeError("image_size values must be integers")
        if any(int(value) <= 0 for value in image_size):
            raise ValueError("image_size values must be positive")
        image_size = (int(image_size[0]), int(image_size[1]))

    def clip_frame(frame: Detection) -> None:
        width, height = image_size or frame.image_size
        if len(frame._labels) == 0:
            frame._slice_keep([], "clip_boxes")
            return
        xyxy = frame._xyxy.copy()
        xyxy[:, 0] = np.maximum(0.0, xyxy[:, 0])
        xyxy[:, 1] = np.maximum(0.0, xyxy[:, 1])
        xyxy[:, 2] = np.minimum(float(width), xyxy[:, 2])
        xyxy[:, 3] = np.minimum(float(height), xyxy[:, 3])
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        keep = np.flatnonzero(valid)
        frame._slice_keep(keep, "clip_boxes", new_xyxy=xyxy[keep])

    _frames(result, clip_frame)


def _remove_invalid_boxes(result: Result, min_area: float = 0) -> None:
    """Remove non-finite, inverted, or undersized boxes."""
    minimum = float(min_area)
    if not math.isfinite(minimum):
        raise ValueError("min_area must be finite")
    if minimum < 0:
        raise ValueError("min_area must be non-negative")

    def clean_frame(frame: Detection) -> None:
        if len(frame._labels) == 0:
            frame._slice_keep([], "filter_boxes")
            return
        xyxy = frame._xyxy
        finite = np.isfinite(xyxy).all(axis=1)
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        valid = finite & (widths > 0) & (heights > 0) & ((widths * heights) >= minimum)
        frame._slice_keep(np.flatnonzero(valid), "filter_boxes")

    _frames(result, clean_frame)


def box_iou(first: Box, second: Box) -> float:
    """Return intersection over union for two boxes."""
    if not isinstance(first, Box) or not isinstance(second, Box):
        raise TypeError("first and second must be Box instances")
    x1, y1 = max(first.xmin, second.xmin), max(first.ymin, second.ymin)
    x2, y2 = min(first.xmax, second.xmax), min(first.ymax, second.ymax)
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    union = first.area + second.area - intersection
    return 0. if union <= 0 else intersection / union


def _suppress(result: Result, iou: float = .5, class_agnostic: bool = False) -> None:
    """Apply score-ordered non-maximum suppression."""
    iou = float(iou)
    if not 0 <= iou <= 1:
        raise ValueError("iou must be between 0 and 1")

    def one(frame: Detection) -> None:
        n = len(frame._labels)
        if n == 0:
            frame._slice_keep([], "nms")
            return
        xyxy = frame._xyxy
        scores = frame._scores
        cids = None if class_agnostic else np.asarray(frame.class_ids, dtype=np.int64)
        if n >= 512 and np.unique(scores).size == n:
            try:
                import torch
                from torchvision.ops import batched_nms, nms as torch_nms

                boxes_tensor = torch.as_tensor(xyxy, dtype=torch.float32)
                scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
                if class_agnostic:
                    kept = torch_nms(boxes_tensor, scores_tensor, iou)
                else:
                    kept = batched_nms(
                        boxes_tensor, scores_tensor,
                        torch.as_tensor(cids, dtype=torch.int64), iou,
                    )
                frame._slice_keep(kept.cpu().numpy(), "nms")
                return
            except (ImportError, OSError, RuntimeError):
                pass
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []
        while order.size > 0:
            i = int(order[0])
            kept.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
            yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
            xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
            yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[rest] - inter
            ovr = np.where(union <= 0.0, 0.0, inter / union)
            if class_agnostic:
                mask = ovr <= iou
            else:
                mask = (cids[rest] != cids[i]) | (ovr <= iou)
            order = rest[mask]
        frame._slice_keep(kept, "nms")

    _frames(result, one)


def _top_boxes(result: Result, count: int) -> None:
    """Keep the highest-scoring boxes per frame."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")

    def top_frame(frame: Detection) -> None:
        if len(frame._labels) == 0:
            frame._slice_keep([], "top")
            return
        order = np.argsort(-frame._scores, kind="stable")[:count]
        frame._slice_keep(order, "top")

    _frames(result, top_frame)


def select(*, where: Optional[Callable[[Box], bool]] = None,
           labels: Optional[str | Iterable[str]] = None,
           class_ids: Optional[int | Iterable[int]] = None,
           uids: Optional[int | Iterable[int]] = None):
    """Build an operation that keeps boxes matching every selector."""
    from .core import Operation
    if where is not None and not callable(where):
        raise TypeError("predicate must be callable")
    labels = None if labels is None else ({labels} if isinstance(labels, str) else set(labels))
    class_ids = None if class_ids is None else (
        {int(class_ids)} if isinstance(class_ids, (int, np.integer)) else set(class_ids)
    )
    uids = None if uids is None else (
        {int(uids)} if isinstance(uids, (int, np.integer)) else set(uids)
    )
    return Operation(
        "select",
        lambda result: _select(
            result, where=where, labels=labels, class_ids=class_ids, uids=uids
        ),
        _trusted=where is None,
    )


def map_boxes(function: Callable[[Box], Optional[Box]]):
    """Build an operation that transforms every box atomically."""
    if not callable(function):
        raise TypeError("function must be callable")
    from .core import Operation
    return Operation("map_boxes", lambda result: _map_boxes(result, function))


def threshold(minimum: float):
    """Build a score-threshold operation."""
    from .core import Operation
    minimum = float(minimum)
    if not math.isfinite(minimum):
        raise ValueError("minimum must be finite")
    return Operation("threshold", lambda result: _filter_score(result, minimum), _trusted=True)


def clip(image_size: Optional[tuple[int, int]] = None):
    """Build a clipping operation."""
    from .core import Operation
    if image_size is not None:
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise TypeError("image_size must be a (width, height) pair")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in image_size):
            raise TypeError("image_size values must be integers")
        if any(int(value) <= 0 for value in image_size):
            raise ValueError("image_size values must be positive")
        image_size = int(image_size[0]), int(image_size[1])
    return Operation("clip", lambda result: _clip_boxes(result, image_size), _trusted=True)


def remove_invalid(min_area: float = 0):
    """Build an invalid-box removal operation."""
    from .core import Operation
    minimum = float(min_area)
    if not math.isfinite(minimum):
        raise ValueError("min_area must be finite")
    if minimum < 0:
        raise ValueError("min_area must be non-negative")
    return Operation("remove_invalid", lambda result: _remove_invalid_boxes(result, minimum), _trusted=True)


def nms(iou: float = .5, class_agnostic: bool = False):
    """Build a non-maximum-suppression operation."""
    from .core import Operation
    iou = float(iou)
    if not 0 <= iou <= 1:
        raise ValueError("iou must be between 0 and 1")
    if not isinstance(class_agnostic, bool):
        raise TypeError("class_agnostic must be a boolean")
    return Operation("nms", lambda result: _suppress(result, iou, class_agnostic), _trusted=True)


def top(count: int):
    """Build a top-score operation."""
    from .core import Operation
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    return Operation("top", lambda result: _top_boxes(result, count), _trusted=True)


__all__ = [
    "select", "map_boxes", "box_iou", "threshold", "clip",
    "remove_invalid", "nms", "top",
]
