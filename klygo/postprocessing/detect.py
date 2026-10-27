"""Backend-neutral, in-place detection postprocessing."""
import math
from typing import Callable, Iterable, Optional

import numpy as np

from klygo.outputs.detect import Box, Detection, Detections, _array_module

Result = Detection | Detections

_CUPY_NMS_KERNELS = {}


def _cupy_nms(module, xyxy, scores, class_ids, iou, class_agnostic):
    """Return score-ordered NMS indices using CuPy CUDA kernels."""
    coordinate_type = "double" if xyxy.dtype == module.float64 else "float"
    kernels = _CUPY_NMS_KERNELS.get(coordinate_type)
    if kernels is None:
        source = f"""
        #define NMS_THREADS 64

        extern "C" __global__ void nms_mask(
            const {coordinate_type}* boxes,
            const long long* classes,
            const int count,
            const int column_blocks,
            const {coordinate_type} threshold,
            const int class_agnostic,
            unsigned long long* mask) {{
            const int row_block = blockIdx.y;
            const int column_block = blockIdx.x;
            if (row_block > column_block) return;

            const int row_start = row_block * NMS_THREADS;
            const int column_start = column_block * NMS_THREADS;
            const int row_size = min(count - row_start, NMS_THREADS);
            const int column_size = min(count - column_start, NMS_THREADS);

            __shared__ {coordinate_type} block_boxes[NMS_THREADS * 4];
            __shared__ long long block_classes[NMS_THREADS];
            if (threadIdx.x < column_size) {{
                const int index = column_start + threadIdx.x;
                block_boxes[threadIdx.x * 4] = boxes[index * 4];
                block_boxes[threadIdx.x * 4 + 1] = boxes[index * 4 + 1];
                block_boxes[threadIdx.x * 4 + 2] = boxes[index * 4 + 2];
                block_boxes[threadIdx.x * 4 + 3] = boxes[index * 4 + 3];
                block_classes[threadIdx.x] = classes[index];
            }}
            __syncthreads();

            if (threadIdx.x < row_size) {{
                const int current = row_start + threadIdx.x;
                const {coordinate_type}* current_box = boxes + current * 4;
                const {coordinate_type} current_area =
                    (current_box[2] - current_box[0]) *
                    (current_box[3] - current_box[1]);
                const int start = row_block == column_block ? threadIdx.x + 1 : 0;
                unsigned long long suppressed = 0;

                for (int index = start; index < column_size; ++index) {{
                    if (!class_agnostic &&
                            classes[current] != block_classes[index]) continue;
                    const {coordinate_type} left =
                        max(current_box[0], block_boxes[index * 4]);
                    const {coordinate_type} top =
                        max(current_box[1], block_boxes[index * 4 + 1]);
                    const {coordinate_type} right =
                        min(current_box[2], block_boxes[index * 4 + 2]);
                    const {coordinate_type} bottom =
                        min(current_box[3], block_boxes[index * 4 + 3]);
                    const {coordinate_type} width =
                        max(({coordinate_type})0, right - left);
                    const {coordinate_type} height =
                        max(({coordinate_type})0, bottom - top);
                    const {coordinate_type} intersection = width * height;
                    const {coordinate_type} other_area =
                        (block_boxes[index * 4 + 2] - block_boxes[index * 4]) *
                        (block_boxes[index * 4 + 3] - block_boxes[index * 4 + 1]);
                    const {coordinate_type} union_area =
                        current_area + other_area - intersection;
                    if (union_area > 0 && intersection / union_area > threshold)
                        suppressed |= 1ULL << index;
                }}
                mask[current * column_blocks + column_block] = suppressed;
            }}
        }}

        extern "C" __global__ void nms_select(
            const unsigned long long* mask,
            const int count,
            const int column_blocks,
            unsigned long long* removed,
            long long* keep,
            int* keep_count) {{
            if (blockIdx.x != 0 || threadIdx.x != 0) return;
            int selected = 0;
            for (int index = 0; index < count; ++index) {{
                const int block = index / NMS_THREADS;
                const int offset = index % NMS_THREADS;
                if (!(removed[block] & (1ULL << offset))) {{
                    keep[selected++] = index;
                    const unsigned long long* row = mask + index * column_blocks;
                    for (int column = block; column < column_blocks; ++column)
                        removed[column] |= row[column];
                }}
            }}
            keep_count[0] = selected;
        }}
        """
        kernels = (
            module.RawKernel(source, "nms_mask"),
            module.RawKernel(source, "nms_select"),
        )
        _CUPY_NMS_KERNELS[coordinate_type] = kernels

    order = module.argsort(-scores, kind="stable")
    ordered_boxes = module.ascontiguousarray(xyxy[order])
    if class_ids is None:
        ordered_classes = module.zeros(len(order), dtype=module.int64)
    else:
        ordered_classes = module.ascontiguousarray(class_ids[order], dtype=module.int64)
    count = len(order)
    column_blocks = (count + 63) // 64
    mask = module.zeros((count, column_blocks), dtype=module.uint64)
    threshold = module.float64(iou) if coordinate_type == "double" else module.float32(iou)
    kernels[0](
        (column_blocks, column_blocks),
        (64,),
        (
            ordered_boxes,
            ordered_classes,
            count,
            column_blocks,
            threshold,
            int(class_agnostic),
            mask,
        ),
    )
    removed = module.zeros(column_blocks, dtype=module.uint64)
    keep = module.empty(count, dtype=module.int64)
    keep_count = module.zeros(1, dtype=module.int32)
    kernels[1]((1,), (1,), (mask, count, column_blocks, removed, keep, keep_count))
    selected = int(keep_count.item())
    return order[keep[:selected]]


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
        module = _array_module(frame.device)
        keep = module.flatnonzero(frame._scores >= minimum)
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
        module = _array_module(frame.device)
        xyxy = frame._xyxy.copy()
        xyxy[:, 0] = module.maximum(0.0, xyxy[:, 0])
        xyxy[:, 1] = module.maximum(0.0, xyxy[:, 1])
        xyxy[:, 2] = module.minimum(float(width), xyxy[:, 2])
        xyxy[:, 3] = module.minimum(float(height), xyxy[:, 3])
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        keep = module.flatnonzero(valid)
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
        module = _array_module(frame.device)
        xyxy = frame._xyxy
        calculation = (
            xyxy.astype(module.float32)
            if module is not np and xyxy.dtype == module.float16
            else xyxy
        )
        finite = module.isfinite(calculation).all(axis=1)
        widths = calculation[:, 2] - calculation[:, 0]
        heights = calculation[:, 3] - calculation[:, 1]
        valid = finite & (widths > 0) & (heights > 0) & ((widths * heights) >= minimum)
        frame._slice_keep(module.flatnonzero(valid), "filter_boxes")

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
        module = _array_module(frame.device)
        calculation_xyxy = (
            xyxy.astype(module.float32)
            if module is not np and xyxy.dtype == module.float16
            else xyxy
        )
        calculation_scores = (
            scores.astype(module.float32)
            if module is not np and scores.dtype == module.float16
            else scores
        )
        cids = None if class_agnostic else frame._class_ids.astype(module.int64, copy=False)
        if module is not np:
            kept = _cupy_nms(
                module,
                calculation_xyxy,
                calculation_scores,
                cids,
                iou,
                class_agnostic,
            )
            frame._slice_keep(kept, "nms")
            return
        xyxy = calculation_xyxy
        scores = calculation_scores
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = module.argsort(-scores, kind="stable")
        kept: list[int] = []
        while order.size > 0:
            i = int(order[0])
            kept.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = module.maximum(xyxy[i, 0], xyxy[rest, 0])
            yy1 = module.maximum(xyxy[i, 1], xyxy[rest, 1])
            xx2 = module.minimum(xyxy[i, 2], xyxy[rest, 2])
            yy2 = module.minimum(xyxy[i, 3], xyxy[rest, 3])
            w = module.maximum(0.0, xx2 - xx1)
            h = module.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[rest] - inter
            ovr = module.where(union <= 0.0, 0.0, inter / union)
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
        module = _array_module(frame.device)
        scores = frame._scores
        if module is not np and scores.dtype == module.float16:
            scores = scores.astype(module.float32)
        order = module.argsort(-scores, kind="stable")[:count]
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
