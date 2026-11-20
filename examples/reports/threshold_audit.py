"""Runnable example for reports.threshold_audit."""

from klygo import outputs, reports

pred = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat", score=0.9)], source_id="a", image_size=(64, 64))])
target = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat")], source_id="a", image_size=(64, 64))])
print(reports.threshold_audit(pred, target, thresholds=[0.3, 0.5])["best_threshold"])
