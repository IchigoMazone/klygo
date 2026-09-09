"""
Ultralytics Backend Logic (klygo.models.backend.ultralytics).
Chứa toàn bộ logic xử lý đặc thù cho các mô hình Ultralytics YOLO.
"""

import os
from typing import Any, List, Dict


def format_results(ultra_results: Any) -> List[Dict[str, Any]]:
    """Bóc tách boxes, scores, labels từ kết quả trả về của Ultralytics YOLO."""
    raw_list = []
    if ultra_results is not None:
        for res in ultra_results:
            boxes_data, scores_data, labels_data = [], [], []
            if getattr(res, "boxes", None) is not None:
                for b in res.boxes:
                    boxes_data.append(b.xyxy[0].tolist())
                    scores_data.append(float(b.conf[0].item()))
                    cls_id = int(b.cls[0].item())
                    labels_data.append(res.names.get(cls_id, str(cls_id)))
            raw_list.append({"boxes": boxes_data, "scores": scores_data, "labels": labels_data})
    return raw_list


def save(model: Any, output_dir: str) -> None:
    """Lưu trọng số theo chuẩn Ultralytics YOLO."""
    abs_out = os.path.abspath(output_dir)
    if model is not None and hasattr(model, "save"):
        model.save(abs_out)
