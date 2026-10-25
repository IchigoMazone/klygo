"""Backend-neutral, in-place detection postprocessing."""
import math
from typing import Callable, Iterable, Mapping, Optional
from klygo.outputs.detect import Box, Detection, Detections

Result = Detection | Detections

def _frames(result: Result, function: Callable[[Detection], None]) -> None:
    if isinstance(result, Detections): result.map(lambda frame: function(frame))
    elif isinstance(result, Detection): function(result)
    else: raise TypeError("result must be a Detection or Detections")

def select(result: Result, *, where: Optional[Callable[[Box], bool]] = None,
           labels: Optional[Iterable[str]] = None, class_ids: Optional[Iterable[int]] = None,
           uids: Optional[Iterable[int]] = None) -> None:
    """Keep boxes matching every supplied selector."""
    ls = None if labels is None else set(labels); cs = None if class_ids is None else set(class_ids); us = None if uids is None else set(uids)
    def matches(box: Box) -> bool:
        return ((where is None or bool(where(box))) and (ls is None or box.label in ls) and
                (cs is None or box.class_id in cs) and (us is None or box.uid in us))
    _frames(result, lambda frame: frame.filter(matches))

def map_boxes(result: Result, function: Callable[[Box], Optional[Box]]) -> None:
    """Transform boxes atomically."""
    _frames(result, lambda frame: frame.map(function))

def relabel(result: Result, mapping: Mapping[str, str]) -> None:
    """Rename or merge labels."""
    if isinstance(result, (Detection, Detections)): result.relabel(mapping)
    else: raise TypeError("result must be a Detection or Detections")

def drop(result: Result, *, labels: Iterable[str]) -> None:
    """Remove boxes with matching labels."""
    if isinstance(result, (Detection, Detections)): result.drop(labels)
    else: raise TypeError("result must be a Detection or Detections")

def filter_score(result: Result, minimum: float) -> None:
    """Remove boxes below a score threshold."""
    minimum = float(minimum); select(result, where=lambda box: box.score >= minimum)

def clip_boxes(result: Result, image_size: Optional[tuple[int, int]] = None) -> None:
    """Clip coordinates and discard boxes wholly outside the image."""
    def clip_frame(frame: Detection) -> None:
        width, height = image_size or frame.image_size; kept = []
        for original in frame:
            values = max(0., original.xmin), max(0., original.ymin), min(float(width), original.xmax), min(float(height), original.ymax)
            if values[2] <= values[0] or values[3] <= values[1]: continue
            item = original._clone(); item.box = values; kept.append(item)
        frame._commit(kept, "clip_boxes")
    _frames(result, clip_frame)

def remove_invalid_boxes(result: Result, min_area: float = 0) -> None:
    """Remove non-finite, inverted, or undersized boxes."""
    minimum = float(min_area)
    if minimum < 0: raise ValueError("min_area must be non-negative")
    select(result, where=lambda box: all(math.isfinite(v) for v in box.box) and box.area >= minimum)

def box_iou(first: Box, second: Box) -> float:
    """Return intersection over union for two boxes."""
    if not isinstance(first, Box) or not isinstance(second, Box): raise TypeError("first and second must be Box instances")
    x1, y1, x2, y2 = max(first.xmin, second.xmin), max(first.ymin, second.ymin), min(first.xmax, second.xmax), min(first.ymax, second.ymax)
    intersection = max(0., x2-x1) * max(0., y2-y1); union = first.area + second.area - intersection
    return 0. if union <= 0 else intersection / union

def suppress(result: Result, iou: float = .5, class_agnostic: bool = False) -> None:
    """Apply score-ordered non-maximum suppression."""
    iou = float(iou)
    if not 0 <= iou <= 1: raise ValueError("iou must be between 0 and 1")
    def one(frame: Detection) -> None:
        remaining = sorted(frame, key=lambda box: box.score, reverse=True); kept = []
        while remaining:
            current = remaining.pop(0); kept.append(current)
            remaining = [candidate for candidate in remaining if (not class_agnostic and candidate.class_id != current.class_id) or box_iou(current, candidate) <= iou]
        frame._commit(kept, "nms")
    _frames(result, one)

def top_boxes(result: Result, count: int) -> None:
    """Keep the highest-scoring boxes per frame."""
    if isinstance(count, bool) or not isinstance(count, int): raise TypeError("count must be an integer")
    if count < 0: raise ValueError("count must be non-negative")
    _frames(result, lambda frame: frame._commit(sorted(frame, key=lambda box: box.score, reverse=True)[:count], "top"))

def threshold(minimum: float):
    """Build a score-threshold operation."""
    from .core import Operation
    return Operation("threshold", lambda result: filter_score(result, minimum))
def clip(image_size: Optional[tuple[int, int]] = None):
    """Build a clipping operation."""
    from .core import Operation
    return Operation("clip", lambda result: clip_boxes(result, image_size))
def remove_invalid(min_area: float = 0):
    """Build an invalid-box removal operation."""
    from .core import Operation
    return Operation("remove_invalid", lambda result: remove_invalid_boxes(result, min_area))
def nms(iou: float = .5, class_agnostic: bool = False):
    """Build a non-maximum-suppression operation."""
    from .core import Operation
    return Operation("nms", lambda result: suppress(result, iou, class_agnostic))
def top(count: int):
    """Build a top-score operation."""
    from .core import Operation
    return Operation("top", lambda result: top_boxes(result, count))

__all__ = ["select", "map_boxes", "relabel", "drop", "filter_score", "clip_boxes", "remove_invalid_boxes", "box_iou", "suppress", "top_boxes", "threshold", "clip", "remove_invalid", "nms", "top"]
