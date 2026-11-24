"""Composition and shared execution of detection metrics."""
from typing import Any
from klygo import outputs, validators
from klygo.utils.progress import ProgressBar
from ._core import (_Metric, _Composition, _Context, _summary, _ap, _cat,
                    _host, _Rows, _MatchReport)

__all__ = ["compose", "evaluate"]


def compose(*metrics) -> _Composition:
    """Group one or more deferred metric configurations without executing them.

    Parameters
    ----------
    *metrics : metric configuration
        Results of metric functions called without pred and target.

    Returns
    -------
    configuration
        Immutable reusable group for evaluate.

    Raises
    ------
    TypeError
        An argument is not a metric configuration.
    ValueError
        The group is empty or contains duplicate metric names.
    """
    if not metrics:
        raise ValueError("compose requires at least one metric")
    for metric in metrics:
        validators.validate_type(metric, _Metric, "metric")
    if len({m.name for m in metrics}) != len(metrics):
        raise ValueError("compose does not accept duplicate metric names")
    return _Composition(tuple(metrics))


def evaluate(pred, target, evaluator=None, *, verbose=False) -> dict[str, Any]:
    """Evaluate matching Detection pairs or materialized Detections collections.

    Parameters
    ----------
    pred, target : Detection or Detections
        Same-type inputs with identical frame identities and image sizes.
    evaluator : configuration, optional
        A compose result. Defaults to precision_recall and map_score.

    verbose : bool, default=False
        Display one shared bar across preparation, matching, and metric completion.

    Returns
    -------
    dict
        Results keyed by the selected metric names.

    Raises
    ------
    TypeError
        Inputs or evaluator have unsupported types.
    ValueError
        Frame identities, sizes, or devices differ, or an input is streaming.
    """
    validators.validate_type(verbose, bool, "verbose")
    if evaluator is None:
        from .operations import precision_recall, map_score
        evaluator = compose(precision_recall(), map_score())
    validators.validate_type(evaluator, _Composition, "evaluator")
    return _execute(pred, target, evaluator.metrics, verbose)


def _execute(pred, target, selected, verbose):
    # Count CPU-side work only; progress never reads or synchronizes GPU arrays.
    matching = set()
    for metric in selected:
        options = dict(metric.options)
        thresholds = (options['thresholds'] if metric.name in {'map_score', 'iou_sweep'}
                      else (options.get('iou', .5),))
        matching.update((threshold, metric.name == 'confusion_matrix') for threshold in thresholds)
    frames = (1 if isinstance(pred, outputs.Detection)
              else len(pred) if isinstance(pred, outputs.Detections) and not pred.is_stream else 0)
    total = frames * (1 + len(matching)) + len(selected)
    with ProgressBar(total, "Evaluating metrics", unit="step", verbose=verbose) as progress:
        if progress.bar is not None:
            progress.bar.mininterval = 0.2
        context = _Context(pred, target, progress=progress if verbose else None)
        with context.scope():
            result = {}
            for metric in selected:
                if progress.bar is not None:
                    progress.bar.set_description(f"Computing {metric.name}", refresh=False)
                result[metric.name] = _run(context, metric)
                progress.update()
            return _host(result, context.xp)


def _pooled(context, iou, cutoff=0):
    xp = context.xp
    tp = xp.asarray(0, dtype=xp.int64)
    fp = xp.asarray(0, dtype=xp.int64)
    support = 0
    overlap_sum = xp.asarray(0., dtype=xp.float64)
    for frame, match in zip(context.frames, context.matches(iou)):
        kept = frame.pred['scores'] >= cutoff
        hit = match.target >= 0
        tp = tp + xp.sum(kept & hit)
        fp = fp + xp.sum(kept & ~hit)
        support += len(frame.target_labels)
        overlap_sum = overlap_sum + xp.sum(match.iou[kept & hit])
    return _summary(tp, fp, support - tp, overlap_sum, xp)


def _class_rows(context, iou):
    if iou in context.class_cache:
        return context.class_cache[iou]
    xp = context.xp
    matches = context.matches(iou)
    rows = []
    for index, label in enumerate(context.labels):
        scores, correct = [], []
        overlap_sum = xp.asarray(0., dtype=xp.float64)
        for frame, match in zip(context.frames, matches):
            selected = frame.pred_labels == index
            scores.append(frame.pred['scores'][selected])
            correct.append(match.target[selected] >= 0)
            overlap_sum = overlap_sum + xp.sum(match.iou[selected])
        scores, correct = _cat(scores, xp), _cat(correct, xp, xp.bool_)
        tp = xp.sum(correct, dtype=xp.int64)
        support = context.support[label]
        rows.append(dict(label=label, support=support, predictions=len(scores),
                         **_summary(tp, len(scores) - tp, support - tp, overlap_sum, xp),
                         ap=_ap(scores, correct, support, xp)))
    context.class_cache[iou] = rows
    return rows


def _area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _size_rows(context, iou):
    xp = context.xp
    matched = context.matches(iou)
    rows = []
    boundaries = xp.asarray([32**2, 96**2], dtype=xp.float64)
    groups = []
    for frame, match in zip(context.frames, matched):
        pred_area = _area(frame.pred['boxes'].astype(xp.float64))
        target_area = _area(frame.target['boxes'].astype(xp.float64))
        hit = match.target >= 0
        assigned = xp.where(hit, target_area[xp.maximum(match.target, 0)], pred_area) if len(target_area) else pred_area
        groups.append((xp.searchsorted(boundaries, assigned, side='right'),
                       xp.searchsorted(boundaries, target_area, side='right')))
    for index, size in enumerate(('small', 'medium', 'large')):
        scores, correct = [], []
        support = xp.asarray(0, dtype=xp.int64)
        overlap_sum = xp.asarray(0., dtype=xp.float64)
        class_scores = [[] for _ in context.labels]
        class_hits = [[] for _ in context.labels]
        class_support = [xp.asarray(0, dtype=xp.int64) for _ in context.labels]
        for frame, match, (pred_group, target_group) in zip(context.frames, matched, groups):
            selected = pred_group == index
            scores.append(frame.pred['scores'][selected])
            correct.append(match.target[selected] >= 0)
            support = support + xp.sum(target_group == index)
            overlap_sum = overlap_sum + xp.sum(match.iou[selected])
            for class_id in range(len(context.labels)):
                mask = selected & (frame.pred_labels == class_id)
                class_scores[class_id].append(frame.pred['scores'][mask])
                class_hits[class_id].append(match.target[mask] >= 0)
                class_support[class_id] = class_support[class_id] + xp.sum((target_group == index) & (frame.target_labels == class_id))
        scores, correct = _cat(scores, xp), _cat(correct, xp, xp.bool_)
        tp = xp.sum(correct, dtype=xp.int64)
        aps = xp.stack([_ap(_cat(s, xp), _cat(h, xp, xp.bool_), n, xp)
                        for s, h, n in zip(class_scores, class_hits, class_support)]) if context.labels else xp.empty(0)
        valid = xp.isfinite(aps)
        count = xp.sum(valid)
        average = xp.where(count > 0, xp.sum(xp.where(valid, aps, 0)) / xp.maximum(count, 1), xp.nan)
        rows.append(dict(size=size, support=support, predictions=len(scores),
                         **_summary(tp, len(scores)-tp, support-tp, overlap_sum, xp), ap=average))
    return rows


def _run(context, metric):
    xp = context.xp
    options = dict(metric.options)
    name = metric.name
    iou = options.get('iou', .5)
    if name == 'precision_recall':
        return _pooled(context, iou)
    if name == 'f1_score':
        summary = _pooled(context, iou)
        precision = summary['precision']
        recall = summary['recall']
        denom = precision + recall
        f1_value = xp.where(denom > 0, 2 * precision * recall / xp.maximum(denom, 1e-12), 0.0)
        return {"f1": f1_value, "precision": precision, "recall": recall}
    if name == 'per_class':
        return _class_rows(context, iou)
    if name == 'map_score':
        thresholds = options['thresholds']
        rows = [_class_rows(context, t) for t in thresholds]
        valid = [i for i, label in enumerate(context.labels) if context.support[label]]
        means = [sum(row[i]['ap'] for i in valid) / len(valid) if valid else None for row in rows]
        classes = [dict(label=label, support=context.support[label],
                        ap=sum(row[i]['ap'] for row in rows) / len(rows),
                        ap_by_iou=[dict(iou=t, ap=row[i]['ap']) for t, row in zip(thresholds, rows)])
                   for i, label in enumerate(context.labels)]
        result = dict(map=sum(means)/len(means) if valid else None, per_class=classes,
                      per_iou=[dict(iou=t, map=value) for t, value in zip(thresholds, means)])
        for t, value in zip(thresholds, means):
            if t in (.5, .75):
                result[f'map_{round(t*100)}'] = value
        return result
    if name == 'iou_sweep':
        return [dict(iou=t, **_pooled(context, t)) for t in options['thresholds']]
    if name == 'threshold_sweep':
        return [dict(threshold=t, **_pooled(context, iou, t)) for t in options['thresholds']]
    if name == 'size_breakdown':
        return _size_rows(context, iou)
    matches = context.matches(iou, name == 'confusion_matrix')
    if name == 'match_boxes':
        reports = []
        for frame, match in zip(context.frames, matches):
            hit = match.target[frame.order] >= 0
            indices = frame.order[hit]
            pairs = xp.stack((indices, match.target[indices], match.iou[indices]), axis=1)
            reports.append(_MatchReport(frame.identity, pairs, frame.order[~hit], xp.flatnonzero(~match.used)))
        return reports
    if name == 'per_image':
        rows = []
        for frame, match in zip(context.frames, matches):
            tp = xp.sum(match.target >= 0)
            rows.append(dict(identity=frame.identity, targets=len(frame.target_labels), predictions=len(frame.pred_labels),
                             **_summary(tp, len(frame.pred_labels)-tp, len(frame.target_labels)-tp, xp.sum(match.iou), xp)))
        return rows
    if name == 'confusion_matrix':
        background = len(context.labels)
        matrix_dtype = xp.int64 if getattr(xp, "__name__", "") == "numpy" else xp.int32
        matrix = xp.zeros((background+1, background+1), dtype=matrix_dtype)
        for frame, match in zip(context.frames, matches):
            hit = match.target >= 0
            xp.add.at(matrix, (frame.target_labels[match.target[hit]], frame.pred_labels[hit]), 1)
            xp.add.at(matrix, (xp.full(len(frame.pred_labels[~hit]), background), frame.pred_labels[~hit]), 1)
            missed = frame.target_labels[~match.used]
            xp.add.at(matrix, (missed, xp.full(len(missed), background)), 1)
        return dict(labels=context.labels+[None], background_index=background, matrix=matrix)
    if name == 'calibration':
        scores = _cat([frame.pred['scores'] for frame in context.frames], xp)
        hit = _cat([match.target >= 0 for match in matches], xp, xp.bool_)
        bins = options['bins']
        indices = xp.minimum((scores*bins).astype(xp.int32), bins-1)
        counts = xp.bincount(indices, minlength=bins)
        sums = xp.bincount(indices, weights=scores, minlength=bins)
        successes = xp.bincount(indices, weights=hit.astype(xp.float64), minlength=bins)
        confidence = xp.where(counts > 0, sums/xp.maximum(counts, 1), xp.nan)
        accuracy = xp.where(counts > 0, successes/xp.maximum(counts, 1), xp.nan)
        ece = xp.sum(xp.abs(sums-successes))/len(scores) if len(scores) else None
        rows = [dict(lower=i/bins, upper=(i+1)/bins, count=counts[i],
                     confidence=confidence[i], accuracy=accuracy[i]) for i in range(bins)]
        return dict(ece=ece, bins=rows, confidence=dict(tp=scores[hit], fp=scores[~hit]))
    if name == 'error_analysis':
        categories = ('duplicate', 'classification', 'localization', 'classification_localization', 'background', 'missed')
        counts = {name: xp.asarray(0, dtype=xp.int64) for name in categories}
        details = {name: [] for name in categories}
        for frame, match in zip(context.frames, matches):
            overlaps = frame.overlaps
            same = frame.pred_labels[:, None] == frame.target_labels[None, :]
            high = overlaps >= iou
            near = (overlaps >= options['bg_iou']) & (overlaps < iou)
            predicates = [xp.any(high & same & match.used[None, :], axis=1),
                          xp.any(high & ~same, axis=1), xp.any(near & same, axis=1),
                          xp.any(near & ~same, axis=1)]
            assigned = xp.full(len(frame.pred_labels), -1, dtype=xp.int32)
            for index, mask in enumerate(predicates):
                assigned = xp.where((assigned < 0) & mask, index, assigned)
            assigned = xp.where(assigned < 0, 4, assigned)
            assigned = xp.where(match.target >= 0, -1, assigned)
            for index, category in enumerate(categories[:-1]):
                # Preserve confidence order in error details, as in match_boxes.
                selected = frame.order[assigned[frame.order] == index]
                counts[category] = counts[category] + len(selected)
                details[category].append(_Rows(frame.identity, 'pred_index', selected))
            missed = xp.flatnonzero(~match.used)
            counts['missed'] = counts['missed'] + len(missed)
            details['missed'].append(_Rows(frame.identity, 'target_index', missed))
        return dict(counts=counts, details=details)
    raise ValueError(f'Unknown metric: {name}')
