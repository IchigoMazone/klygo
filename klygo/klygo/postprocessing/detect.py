"""Pythonic transformations for ``Box``, ``Detection`` and ``Detections``."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from klygo.outputs.detect import Box, Detection, Detections
from .core import Operation

Result = Union[Detection, Detections]
Predicate = Callable[[Box], bool]


def _copy_detection(result: Detection, boxes: Iterable[Box]) -> Detection:
    copied = Detection(
        result.source_image,
        list(boxes),
        dict(result.speed),
        result.image_frame_index,
        config=dict(result.config),
    )
    copied.timestamp = result.timestamp
    copied._source_path = result.source_path
    copied._url = result.url
    return copied


def _map_result(result: Result, function: Callable[[Detection], Detection]) -> Result:
    if isinstance(result, Detections):
        return result.map(function)
    if isinstance(result, Detection):
        return function(result)
    raise TypeError("result must be Detection or Detections")


def _matches(
    box: Box,
    *,
    ids: Optional[Iterable[int]] = None,
    uids: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    where: Optional[Predicate] = None,
) -> bool:
    if ids is not None and box.id not in set(ids):
        return False
    if uids is not None and box.uid not in set(uids):
        return False
    if labels is not None and box.label not in set(labels):
        return False
    return where(box) if where is not None else True


def select(
    result: Result,
    *,
    ids: Optional[Iterable[int]] = None,
    uids: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    where: Optional[Predicate] = None,
) -> Result:
    """Keep boxes matching all supplied selectors."""
    id_set = set(ids) if ids is not None else None
    uid_set = set(uids) if uids is not None else None
    label_set = set(labels) if labels is not None else None
    return _map_result(
        result,
        lambda frame: _copy_detection(
            frame,
            (
                box
                for box in frame
                if _matches(box, ids=id_set, uids=uid_set, labels=label_set, where=where)
            ),
        ),
    )


def map_boxes(result: Result, function: Callable[[Box], Optional[Box]]) -> Result:
    """Transform every box; returning ``None`` drops that box."""
    if not callable(function):
        raise TypeError("function must be callable")

    def transform(frame: Detection) -> Detection:
        mapped = []
        for box in frame:
            value = function(box)
            if value is None:
                continue
            if not isinstance(value, Box):
                raise TypeError("map_boxes function must return Box or None")
            mapped.append(value)
        return _copy_detection(frame, mapped)

    return _map_result(result, transform)


def relabel(
    result: Result,
    label: Union[str, Mapping[int, str], Callable[[Box], str]],
    *,
    ids: Optional[Iterable[int]] = None,
    uids: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    where: Optional[Predicate] = None,
) -> Result:
    """Relabel selected boxes while retaining every unselected box."""
    id_set = set(ids) if ids is not None else None
    uid_set = set(uids) if uids is not None else None
    label_set = set(labels) if labels is not None else None

    def rename(box: Box) -> Box:
        if not _matches(box, ids=id_set, uids=uid_set, labels=label_set, where=where):
            return box
        if isinstance(label, Mapping):
            new_label = label.get(box.id, box.label)
        elif callable(label):
            new_label = label(box)
        else:
            new_label = label
        return box.with_label(str(new_label))

    return map_boxes(result, rename)


def drop(
    result: Result,
    *,
    ids: Optional[Iterable[int]] = None,
    uids: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    where: Optional[Predicate] = None,
) -> Result:
    """Remove boxes matching all supplied selectors."""
    id_set = set(ids) if ids is not None else None
    uid_set = set(uids) if uids is not None else None
    label_set = set(labels) if labels is not None else None
    return map_boxes(
        result,
        lambda box: None
        if _matches(box, ids=id_set, uids=uid_set, labels=label_set, where=where)
        else box,
    )


def filter_score(result: Result, minimum: float) -> Result:
    """Keep boxes whose score is at least ``minimum``."""
    threshold_value = float(minimum)
    return select(result, where=lambda box: box.score >= threshold_value)


def clip_boxes(result: Result, image_size: Optional[Tuple[int, int]] = None) -> Result:
    """Clip coordinates to the image bounds."""
    def transform(frame: Detection) -> Detection:
        size = image_size or frame.image_size
        return _copy_detection(frame, (box.clip(size) for box in frame))
    return _map_result(result, transform)


def remove_invalid_boxes(result: Result, min_area: float = 0) -> Result:
    """Remove non-finite, inverted or undersized boxes."""
    minimum = float(min_area)

    def valid(box: Box) -> bool:
        return (
            len(box.box) == 4
            and all(math.isfinite(value) for value in box.box)
            and box.xmax > box.xmin
            and box.ymax > box.ymin
            and box.area >= minimum
        )

    return select(result, where=valid)


def box_iou(first: Union[Box, Sequence[float]], second: Union[Box, Sequence[float]]) -> float:
    """Compute IoU for two XYXY boxes."""
    a = first.box if isinstance(first, Box) else first
    b = second.box if isinstance(second, Box) else second
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress(result: Result, iou: float = 0.5, class_agnostic: bool = False) -> Result:
    """Apply score-ordered non-maximum suppression."""
    threshold_value = float(iou)
    if not 0 <= threshold_value <= 1:
        raise ValueError("iou must be between 0 and 1")

    def transform(frame: Detection) -> Detection:
        remaining = sorted(frame.objects, key=lambda box: box.score, reverse=True)
        kept = []
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            remaining = [
                candidate
                for candidate in remaining
                if (not class_agnostic and candidate.label != current.label)
                or box_iou(current, candidate) <= threshold_value
            ]
        return _copy_detection(frame, kept)

    return _map_result(result, transform)


def top_boxes(result: Result, count: int) -> Result:
    """Keep the highest-scoring ``count`` boxes per frame."""
    limit = int(count)
    if limit < 0:
        raise ValueError("count must be greater than or equal to 0")
    return _map_result(
        result,
        lambda frame: _copy_detection(
            frame,
            sorted(frame.objects, key=lambda box: box.score, reverse=True)[:limit],
        ),
    )


def threshold(minimum: float) -> Operation:
    return Operation("threshold", lambda result: filter_score(result, minimum))


def clip(image_size: Optional[Tuple[int, int]] = None) -> Operation:
    return Operation("clip", lambda result: clip_boxes(result, image_size))


def remove_invalid(min_area: float = 0) -> Operation:
    return Operation("remove_invalid", lambda result: remove_invalid_boxes(result, min_area))


def nms(iou: float = 0.5, class_agnostic: bool = False) -> Operation:
    return Operation("nms", lambda result: suppress(result, iou, class_agnostic))


def top(count: int) -> Operation:
    return Operation("top", lambda result: top_boxes(result, count))


# Short compatibility aliases for early pipeline drafts.
valid = remove_invalid
limit = top


__all__ = [
    "select", "map_boxes", "relabel", "drop", "filter_score", "clip_boxes",
    "remove_invalid_boxes", "box_iou", "suppress", "top_boxes", "threshold",
    "clip", "remove_invalid", "nms", "top", "valid", "limit",
]
