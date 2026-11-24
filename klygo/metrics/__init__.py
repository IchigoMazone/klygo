"""Composable, backend-neutral object-detection evaluation.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/metrics/Document_Metrics_Overview.ipynb

Public API (14 functions across 2 submodules):
    Direct & deferred metrics (`operations.py`):
        1. precision_recall(...)  Pooled precision, recall, F1, and matched mean IoU.
        2. f1_score(...)          F1 score computed from precision and recall.
        3. map_score(...)         Class-wise 101-point interpolated AP and mAP.
        4. confusion_matrix(...)  Class-by-class prediction vs target confusion counts.
        5. match_boxes(...)       Per-box TP/FP/FN matching report.
        6. per_class(...)         Per-class precision, recall, F1, and AP breakdown.
        7. per_image(...)         Per-image matching and score summary.
        8. error_analysis(...)    Categorize localization, classification, duplicate, and background errors.
        9. size_breakdown(...)    Evaluate metrics across small, medium, and large object scales.
       10. calibration(...)       Confidence calibration and Expected Calibration Error (ECE).
       11. threshold_sweep(...)   Sweep confidence thresholds to inspect precision-recall tradeoffs.
       12. iou_sweep(...)         Sweep IoU thresholds to inspect pooled precision, recall, F1, and mean IoU.

    Composition & shared execution (`evaluation.py`):
       13. compose(...)           Group deferred metric configurations into one evaluator.
       14. evaluate(...)          Run one or more metrics over Detection or Detections.
"""
from .operations import *
from .operations import __all__ as _operations
from .evaluation import compose, evaluate

__all__ = [*_operations, "compose", "evaluate"]
