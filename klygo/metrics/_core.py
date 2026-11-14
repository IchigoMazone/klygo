"""Shared validation, matching, and aggregation for detection metrics."""
from dataclasses import dataclass
from numbers import Real
import math
from contextlib import nullcontext
import numpy as np
from klygo import outputs, validators
from klygo.outputs.detect import _array_module

_MISSING = object()


def _unit(value, name):
    validators.validate_type(value, Real, name)
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _thresholds(values, default):
    values = tuple(default if values is None else values)
    result = tuple(_unit(value, "threshold") for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("thresholds must be non-empty and contain no duplicates")
    return tuple(sorted(result))


@dataclass(frozen=True)
class _Metric:
    name: str
    options: tuple


@dataclass(frozen=True)
class _Composition:
    metrics: tuple


def _dispatch(name, pred, target, **options):
    metric = _Metric(name, tuple(options.items()))
    if pred is _MISSING and target is _MISSING:
        return metric
    if pred is _MISSING or target is _MISSING:
        raise TypeError("pred and target must be supplied together")
    from .evaluation import _run
    context = _Context(pred, target)
    with context.scope():
        return _host(_run(context, metric), context.xp)



_GREEDY_SOURCE = r"""
extern "C" __global__ void greedy(
    const double* overlaps, const long long* order,
    const int* pred_labels, const int* target_labels,
    const int n, const int m, const double threshold, const int agnostic,
    int* matches, double* values, unsigned char* used) {
    __shared__ double best_values[128];
    __shared__ int best_indices[128];
    const int lane = threadIdx.x;
    for (int rank = 0; rank < n; ++rank) {
        const int p = (int)order[rank];
        int best = -1;
        double best_iou = -1.0;
        for (int t = lane; t < m; t += blockDim.x) {
            if (used[t] || (!agnostic && pred_labels[p] != target_labels[t])) continue;
            const double value = overlaps[(long long)p * m + t];
            if (value >= threshold && (value > best_iou || (value == best_iou && (best < 0 || t < best)))) {
                best = t; best_iou = value;
            }
        }
        best_values[lane] = best_iou;
        best_indices[lane] = best;
        __syncthreads();
        for (int offset = 64; offset > 0; offset /= 2) {
            if (lane < offset) {
                const double other_value = best_values[lane + offset];
                const int other_index = best_indices[lane + offset];
                if (other_value > best_values[lane] ||
                    (other_value == best_values[lane] && other_index >= 0 &&
                     (best_indices[lane] < 0 || other_index < best_indices[lane]))) {
                    best_values[lane] = other_value;
                    best_indices[lane] = other_index;
                }
            }
            __syncthreads();
        }
        if (lane == 0 && best_indices[0] >= 0) {
            const int target = best_indices[0];
            matches[p] = target; values[p] = best_values[0]; used[target] = 1;
        }
        __syncthreads();
    }
}

extern "C" __global__ void suffix_max(const double* values, double* output, const int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double best = 0.0;
    for (int i = n - 1; i >= 0; --i) { if (values[i] > best) best = values[i]; output[i] = best; }
}
"""
_KERNELS = {}


def _kernel(xp, name):
    key = (xp.cuda.runtime.getDevice(), name)
    if key not in _KERNELS:
        _KERNELS[key] = xp.RawKernel(_GREEDY_SOURCE, name)
    return _KERNELS[key]


def _iou_matrix(first, second, xp):
    first, second = first.astype(xp.float64), second.astype(xp.float64)
    lower = xp.maximum(first[:, None, :2], second[None, :, :2])
    upper = xp.minimum(first[:, None, 2:], second[None, :, 2:])
    extent = xp.maximum(upper - lower, 0)
    intersection = extent[..., 0] * extent[..., 1]
    a = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    b = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = a[:, None] + b[None, :] - intersection
    return intersection / xp.maximum(union, xp.finfo(xp.float64).tiny)


@dataclass
class _Frame:
    identity: tuple
    pred: dict
    target: dict
    pred_labels: object
    target_labels: object
    overlaps: object
    order: object


@dataclass
class _Match:
    target: object
    iou: object
    used: object


@dataclass
class _Rows:
    identity: tuple
    key: str
    indices: object


@dataclass
class _MatchReport:
    identity: tuple
    pairs: object
    extra: object
    missed: object


class _Context:
    def __init__(self, pred, target):
        validators.validate_type(pred, (outputs.Detection, outputs.Detections), "pred")
        validators.validate_type(target, (outputs.Detection, outputs.Detections), "target")
        if isinstance(pred, outputs.Detection) != isinstance(target, outputs.Detection):
            raise TypeError("pred and target must both be Detection or both be Detections")
        if isinstance(pred, outputs.Detections) and (pred.is_stream or target.is_stream):
            raise ValueError("metrics require materialized Detections, not streams")
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")
        self.device = pred.device
        self.xp = _array_module(self.device)
        predictions = [pred] if isinstance(pred, outputs.Detection) else list(pred)
        targets = [target] if isinstance(target, outputs.Detection) else list(target)
        pmap = {frame.identity: frame for frame in predictions}
        tmap = {frame.identity: frame for frame in targets}
        if len(pmap) != len(predictions) or len(tmap) != len(targets):
            raise ValueError("frame identities must be unique")
        if pmap.keys() != tmap.keys():
            raise ValueError("pred and target must contain the same frame identities")
        for identity, frame in pmap.items():
            if frame.image_size != tmap[identity].image_size:
                raise ValueError("matching frames must have the same image_size")
        self.labels = sorted({label for frame in predictions + targets for label in frame.labels})
        label_ids = {label: index for index, label in enumerate(self.labels)}
        self.support = {label: sum(frame.labels.count(label) for frame in targets) for label in self.labels}
        self.frames = []
        with self.scope():
            xp = self.xp
            for identity, p in pmap.items():
                a, b = p.to_arrays(), tmap[identity].to_arrays()
                a['scores'] = a['scores'].astype(xp.float64)
                b['scores'] = b['scores'].astype(xp.float64)
                p_labels = xp.asarray([label_ids[label] for label in a['labels']], dtype=xp.int32)
                t_labels = xp.asarray([label_ids[label] for label in b['labels']], dtype=xp.int32)
                order = xp.lexsort(xp.stack((xp.arange(len(p_labels)), -a['scores']))).astype(xp.int64)
                self.frames.append(_Frame(identity, a, b, p_labels, t_labels,
                                          _iou_matrix(a['boxes'], b['boxes'], xp), order))
        self.cache = {}
        self.class_cache = {}

    def scope(self):
        return nullcontext() if self.device == 'cpu' else self.xp.cuda.Device(int(self.device.split(':')[1]))

    def matches(self, iou, agnostic=False):
        key = (iou, agnostic)
        if key not in self.cache:
            xp = self.xp
            rows = []
            for frame in self.frames:
                n, m = len(frame.pred_labels), len(frame.target_labels)
                matched = xp.full(n, -1, dtype=xp.int32)
                values = xp.zeros(n, dtype=xp.float64)
                used = xp.zeros(m, dtype=xp.uint8)
                if n and m:
                    if xp is np:
                        for pi in frame.order:
                            candidates = np.flatnonzero((used == 0) & (frame.overlaps[pi] >= iou) &
                                (True if agnostic else frame.pred_labels[pi] == frame.target_labels))
                            if len(candidates):
                                ti = candidates[np.argmax(frame.overlaps[pi, candidates])]
                                matched[pi], values[pi], used[ti] = ti, frame.overlaps[pi, ti], 1
                    else:
                        _kernel(xp, 'greedy')((1,), (128,), (frame.overlaps, frame.order,
                            frame.pred_labels, frame.target_labels, np.int32(n), np.int32(m),
                            np.float64(iou), np.int32(agnostic), matched, values, used))
                rows.append(_Match(matched, values, used.astype(xp.bool_)))
            self.cache[key] = rows
        return self.cache[key]


def _summary(tp, fp, fn, overlap_sum, xp):
    precision = tp / xp.maximum(tp + fp, 1)
    recall = tp / xp.maximum(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * tp / xp.maximum(2 * tp + fp + fn, 1),
                mean_iou=overlap_sum / xp.maximum(tp, 1))


def _ap(scores, correct, support, xp):
    support = xp.asarray(support)
    if not len(scores):
        return xp.where(support > 0, 0.0, xp.nan)
    order = xp.lexsort(xp.stack((xp.arange(len(scores)), -scores)))
    true_positives = xp.cumsum(correct[order], dtype=xp.float64)
    recall = true_positives / xp.maximum(support, 1)
    precision = true_positives / xp.arange(1, len(scores) + 1)
    if xp is np:
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
    else:
        envelope = xp.empty_like(precision)
        _kernel(xp, 'suffix_max')((1,), (1,), (precision, envelope, np.int32(len(precision))))
    indices = xp.searchsorted(recall, xp.arange(101, dtype=xp.float64) / 100, side='left')
    sampled = xp.where(indices < len(scores), envelope[xp.minimum(indices, len(scores) - 1)], 0.0)
    return xp.where(support > 0, sampled.mean(), xp.nan)


def _cat(arrays, xp, dtype=None):
    return xp.concatenate(arrays) if arrays else xp.empty(0, dtype=dtype or xp.float64)


def _host_values(value):
    if isinstance(value, _Rows):
        return [dict(identity=value.identity, **{value.key: int(i)}) for i in _host_values(value.indices)]
    if isinstance(value, _MatchReport):
        return dict(identity=value.identity,
                    matches=[dict(pred_index=int(p), target_index=int(t), iou=float(v)) for p, t, v in _host_values(value.pairs)],
                    unmatched_pred=_host_values(value.extra), unmatched_target=_host_values(value.missed))
    if isinstance(value, dict):
        return {key: _host_values(item) for key, item in value.items()}
    if isinstance(value, list) and value and all(isinstance(item, _Rows) for item in value):
        return [row for item in value for row in _host_values(item)]
    if isinstance(value, (list, tuple)):
        return type(value)(_host_values(item) for item in value)
    if hasattr(value, 'get') and type(value).__module__.split('.')[0] == 'cupy':
        value = value.get()
    if isinstance(value, np.ndarray):
        return _host_values(value.tolist())
    if isinstance(value, np.generic):
        return _host_values(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass
class _HostValue:
    index: int


def _host(value, xp):
    """Transfer numeric report leaves in dtype batches, not one scalar at a time."""
    if xp is np:
        return _host_values(value)
    leaves = []

    def walk(item, transform):
        if isinstance(item, _Rows):
            return _Rows(item.identity, item.key, walk(item.indices, transform))
        if isinstance(item, _MatchReport):
            return _MatchReport(item.identity, walk(item.pairs, transform),
                                walk(item.extra, transform), walk(item.missed, transform))
        if isinstance(item, dict):
            return {key: walk(child, transform) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(walk(child, transform) for child in item)
        return transform(item)

    def collect(item):
        if isinstance(item, xp.ndarray):
            leaves.append(item)
            return _HostValue(len(leaves) - 1)
        return item

    template = walk(value, collect)
    groups = {}
    for index, array in enumerate(leaves):
        groups.setdefault(array.dtype.str, []).append((index, array))
    resolved = {}
    for arrays in groups.values():
        flat = xp.concatenate([array.ravel() for _, array in arrays]).get()
        offset = 0
        for index, array in arrays:
            resolved[index] = flat[offset:offset + array.size].reshape(array.shape)
            offset += array.size
    return _host_values(walk(template, lambda item: resolved[item.index] if isinstance(item, _HostValue) else item))
