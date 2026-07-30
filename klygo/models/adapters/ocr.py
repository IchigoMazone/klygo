from typing import Any, List
from .base import BaseAdapter
from ..registry import registry
from klygo import media

class TextRegion:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, text: str, score: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.text = text
        self.score = score

    def __repr__(self):
        return f"[{self.text} -> ({int(self.xmin)}, {int(self.ymin)}, {int(self.xmax)}, {int(self.ymax)})]"

class OCRResult:
    def __init__(self, text: str, regions: List[TextRegion]):
        self.text = text
        self.regions = regions

    def __repr__(self):
        return f"<OCRResult: text='{self.text[:30]}...' regions={len(self.regions)}>"

@registry.register_adapter("ocr")
class OCRAdapter(BaseAdapter):
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

    def read_text(self, source: Any, **kwargs) -> Any:
        res = self.predict(source, **kwargs)
        if isinstance(res, list):
            return [r.text for r in res]
        return res.text

    def _post_process(self, raw_outputs: Any, images: list) -> List[OCRResult]:
        if raw_outputs is None:
            return [OCRResult("", []) for _ in images]
            
        batch_results = []
        for first in raw_outputs:
            full_text = first.get("text", "")
            raw_regions = first.get("regions", [])
            
            regions = []
            for region in raw_regions:
                box = region.get("box", [0, 0, 0, 0])
                regions.append(TextRegion(
                    xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3],
                    text=region.get("text", ""),
                    score=region.get("score", 1.0)
                ))
            batch_results.append(OCRResult(full_text, regions))
            
        return batch_results
