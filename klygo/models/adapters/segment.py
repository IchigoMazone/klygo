from typing import Any, List
from .base import BaseAdapter
from ..registry import registry
from klygo import media
import numpy as np

class SegmentedObject:
    def __init__(self, label: str, score: float, bbox: List[float], mask: np.ndarray):
        self.label = label
        self.score = score
        self.bbox = bbox
        self.mask = mask

class SegmentationResult:
    def __init__(self, objects: List[SegmentedObject]):
        self.objects = objects

    def __repr__(self):
        return f"<SegmentationResult: segments={len(self.objects)}>"

@registry.register_adapter("segment")
class SegmentAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> Any:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        import os
        is_batch = isinstance(source, list) or (isinstance(source, str) and os.path.isdir(source))
        
        images = media.load(source)
        raw_outputs = self.backend.predict(images, **run_params)
        results = self._post_process(raw_outputs, images)
        return results if is_batch else results[0]

    def mask(self, source: Any, **kwargs) -> Any:
        res = self.predict(source, **kwargs)
        if isinstance(res, list):
            return [[obj.mask for obj in r.objects] for r in res]
        return [obj.mask for obj in res.objects]

    def _post_process(self, raw_outputs: Any, images: list) -> List[SegmentationResult]:
        if raw_outputs is None:
            return [SegmentationResult([]) for _ in images]
            
        batch_results = []
        if isinstance(raw_outputs, list) and hasattr(raw_outputs[0], "masks"):
            for result in raw_outputs:
                objects = []
                names = result.names
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()  # Binary masks of shape [N, H, W]
                    boxes = result.boxes
                    for idx, box in enumerate(boxes):
                        coords = box.xyxy[0].tolist()
                        score = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = names[class_id]
                        mask_data = masks[idx]
                        objects.append(SegmentedObject(label, score, coords, mask_data))
                batch_results.append(SegmentationResult(objects))
        else:
            batch_results = [SegmentationResult([]) for _ in images]
            
        return batch_results
