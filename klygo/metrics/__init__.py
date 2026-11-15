"""Composable, backend-neutral object-detection evaluation.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/metrics/Document_Metrics_Overview.ipynb

Public API (13 functions across 2 submodules):
    Direct & deferred metrics (`operations.py`):
        1. precision_recall(...)  Pooled precision, recall, F1, and matched mean IoU.
        2. map_score(...)         Class-wise 101-point interpolated AP and mAP.
        3. confusion_matrix(...)  Class-by-class prediction vs target confusion counts.
        4. match_boxes(...)       Per-box TP/FP/FN matching report.
        5. per_class(...)         Per-class precision, recall, F1, and AP breakdown.
        6. per_image(...)         Per-image matching and score summary.
        7. error_analysis(...)    Categorize localization, classification, duplicate, and background errors.
        8. size_breakdown(...)    Evaluate metrics across small, medium, and large object scales.
        9. calibration(...)       Confidence calibration and Expected Calibration Error (ECE).
       10. threshold_sweep(...)   Sweep confidence thresholds to inspect precision-recall tradeoffs.
       11. iou_sweep(...)         Sweep IoU thresholds across AP curves.

    Composition & shared execution (`evaluation.py`):
       12. compose(...)           Group deferred metric configurations into one evaluator.
       13. evaluate(...)          Run one or more metrics over Detection/Detections or DetectionGroup.
"""
from .operations import *
from .operations import __all__ as _operations
from .evaluation import compose, evaluate

__all__ = [*_operations, "compose", "evaluate"]
