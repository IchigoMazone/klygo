"""Runnable example for reports.rank_evaluations."""

from klygo import reports

print(reports.rank_evaluations([{"code": "S1", "mAP50-95": 0.57}, {"code": "S4", "mAP50-95": 0.60}], baseline="S1"))
