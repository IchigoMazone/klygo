"""Runnable example for reports.mining_breakdown."""

from klygo import outputs, reports

sub = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat", score=0.8)], source_id="a", image_size=(64, 64))])
print(reports.mining_breakdown({"S4": sub}))
