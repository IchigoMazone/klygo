"""Direct and deferred object-detection metric configurations."""
from numbers import Integral
from typing import Any
from klygo import validators
from ._core import _MISSING, _Metric, _dispatch, _unit, _thresholds

__all__ = ["precision_recall", "map_score", "confusion_matrix", "match_boxes",
           "per_class", "per_image", "error_analysis", "size_breakdown",
           "calibration", "threshold_sweep", "iou_sweep"]


def precision_recall(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Compute pooled precision, recall, F1, counts, and matched mean IoU.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/precision_recall.md for the exact output schema and conventions.
    """
    return _dispatch("precision_recall", pred, target, iou=_unit(iou, "iou"))


def map_score(pred=_MISSING, target=_MISSING, *, thresholds=None) -> dict[str, Any] | list[Any] | _Metric:
    """Compute class-wise 101-point interpolated AP and mean AP.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    thresholds : iterable of float, optional
        Unique thresholds in [0, 1]. None selects the documented default grid.

    Returns
    -------
    dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/map_score.md for the exact output schema and conventions.
    """
    return _dispatch("map_score", pred, target,
                     thresholds=_thresholds(thresholds, [round(.5 + .05*i, 2) for i in range(10)]))


def confusion_matrix(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Compute geometry-matched counts with GT rows and prediction columns.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/confusion_matrix.md for the exact output schema and conventions.
    """
    return _dispatch("confusion_matrix", pred, target, iou=_unit(iou, "iou"))


def match_boxes(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Return frame-local matched indices and unmatched prediction/target indices.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/match_boxes.md for the exact output schema and conventions.
    """
    return _dispatch("match_boxes", pred, target, iou=_unit(iou, "iou"))


def per_class(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Return counts, precision, recall, F1, mean IoU, and AP per label.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/per_class.md for the exact output schema and conventions.
    """
    return _dispatch("per_class", pred, target, iou=_unit(iou, "iou"))


def per_image(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Return counts, precision, recall, F1, and mean IoU per frame.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/per_image.md for the exact output schema and conventions.
    """
    return _dispatch("per_image", pred, target, iou=_unit(iou, "iou"))


def error_analysis(pred=_MISSING, target=_MISSING, *, iou=0.5, bg_iou=0.1) -> dict[str, Any] | list[Any] | _Metric:
    """Partition false positives into five error types and list missed GT boxes.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.
    bg_iou : float, optional
        Background boundary in [0, iou). Default is 0.1.

    Returns
    -------
    dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/error_analysis.md for the exact output schema and conventions.
    """
    iou, bg_iou = _unit(iou, "iou"), _unit(bg_iou, "bg_iou")
    if bg_iou >= iou:
        raise ValueError("bg_iou must be smaller than iou")
    return _dispatch("error_analysis", pred, target, iou=iou, bg_iou=bg_iou)


def size_breakdown(pred=_MISSING, target=_MISSING, *, iou=0.5) -> dict[str, Any] | list[Any] | _Metric:
    """Return small/medium/large metrics using COCO area boundaries.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/size_breakdown.md for the exact output schema and conventions.
    """
    return _dispatch("size_breakdown", pred, target, iou=_unit(iou, "iou"))


def calibration(pred=_MISSING, target=_MISSING, *, iou=0.5, bins=10) -> dict[str, Any] | list[Any] | _Metric:
    """Return confidence-bin statistics, TP/FP scores, and detection ECE.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.
    bins : int, optional
        Positive number of equal-width confidence bins. Default is 10.

    Returns
    -------
    dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/calibration.md for the exact output schema and conventions.
    """
    validators.validate_type(bins, Integral, "bins")
    if isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    return _dispatch("calibration", pred, target, iou=_unit(iou, "iou"), bins=int(bins))


def threshold_sweep(pred=_MISSING, target=_MISSING, *, iou=0.5, thresholds=None) -> dict[str, Any] | list[Any] | _Metric:
    """Return pooled metrics after retaining scores at or above each threshold.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    iou : float, optional
        Matching IoU in [0, 1]. Default is 0.5.
    thresholds : iterable of float, optional
        Unique thresholds in [0, 1]. None selects the documented default grid.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/threshold_sweep.md for the exact output schema and conventions.
    """
    return _dispatch("threshold_sweep", pred, target, iou=_unit(iou, "iou"),
                     thresholds=_thresholds(thresholds, [i/20 for i in range(21)]))


def iou_sweep(pred=_MISSING, target=_MISSING, *, thresholds=None) -> dict[str, Any] | list[Any] | _Metric:
    """Return pooled metrics at each requested IoU threshold.

    Parameters
    ----------
    pred, target : Detection or Detections, optional
        Same-type, same-source inputs. Supply both to execute; omit both to configure.
    thresholds : iterable of float, optional
        Unique thresholds in [0, 1]. None selects the documented default grid.

    Returns
    -------
    list of dict or configuration
        Metric results for direct calls, or an immutable deferred configuration.

    Raises
    ------
    TypeError
        Inputs or parameter types are unsupported, or only one input is supplied.
    ValueError
        Parameters, frame identities, image sizes, devices, or stream inputs are invalid.

    Notes
    -----
    See docs/metrics/api/iou_sweep.md for the exact output schema and conventions.
    """
    return _dispatch("iou_sweep", pred, target,
                     thresholds=_thresholds(thresholds, [round(.5 + .05*i, 2) for i in range(10)]))
