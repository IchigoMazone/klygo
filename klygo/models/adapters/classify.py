import os
from typing import Any, List, Dict
from .base import BaseAdapter
from ..registry import registry
from klygo import media
from klygo import files
from klygo.utils.progress import ProgressBar

class ClassificationResult:
    def __init__(self, label: str, score: float, topk: List[Dict[str, Any]]):
        self.label = label
        self.score = score
        self.topk = topk

    def __repr__(self):
        return f"<ClassificationResult: label='{self.label}' score={self.score:.2f}>"

@registry.register_adapter("classify")
class ClassifyAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> Any:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        import torch
        is_tensor = isinstance(source, torch.Tensor) or (isinstance(source, list) and len(source) > 0 and isinstance(source[0], torch.Tensor))
        is_batch = isinstance(source, list) or (isinstance(source, str) and os.path.isdir(source))
        
        if is_tensor:
            if isinstance(source, torch.Tensor) and source.ndim > 1:
                is_batch = True
            raw_outputs = self.backend.predict(source, **run_params)
            images = source if is_batch else [source]
        else:
            images = media.load(source)
            raw_outputs = self.backend.predict(images, **run_params)
            
        results = self._post_process(raw_outputs, images)
        return results if is_batch else results[0]

    def top1(self, source: Any, **kwargs) -> Any:
        res = self.predict(source, **kwargs)
        if isinstance(res, list):
            return [{"label": r.label, "score": r.score} for r in res]
        return {"label": res.label, "score": res.score}

    def top5(self, source: Any, **kwargs) -> Any:
        res = self.predict(source, **kwargs)
        if isinstance(res, list):
            return [r.topk[:5] for r in res]
        return res.topk[:5]

    def _post_process(self, raw_outputs: Any, images: list) -> List[ClassificationResult]:
        if raw_outputs is None:
            return [ClassificationResult("unknown", 0.0, []) for _ in images]
            
        batch_results = []
        
        # Case A: Hugging Face pipeline output list
        if isinstance(raw_outputs, list) and isinstance(raw_outputs[0], list) and isinstance(raw_outputs[0][0], dict) and "label" in raw_outputs[0][0]:
            for out in raw_outputs:
                topk = [{"label": item["label"], "score": item["score"]} for item in out]
                batch_results.append(ClassificationResult(topk[0]["label"], topk[0]["score"], topk))
                
        # Case B: Ultralytics classification results
        elif hasattr(raw_outputs, "probs") or (isinstance(raw_outputs, list) and hasattr(raw_outputs[0], "probs")):
            raw_list = raw_outputs if isinstance(raw_outputs, list) else [raw_outputs]
            for first in raw_list:
                probs = first.probs
                top5_indices = probs.top5
                top5_scores = probs.top5conf.tolist()
                names = first.names
                topk = [{"label": names[idx], "score": score} for idx, score in zip(top5_indices, top5_scores)]
                batch_results.append(ClassificationResult(topk[0]["label"], topk[0]["score"], topk))

        # Case C: PyTorch raw logits / tensor output
        else:
            import torch
            if isinstance(raw_outputs, list):
                logits = torch.tensor(raw_outputs)
            else:
                logits = raw_outputs
                
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
                
            classes = self.config.get("classes", None)
            
            for i in range(logits.shape[0]):
                probs = torch.softmax(logits[i], dim=-1).tolist()
                resolved_classes = classes if classes else [str(j) for j in range(len(probs))]
                
                topk = []
                for idx, score in enumerate(probs):
                    if idx < len(resolved_classes):
                        topk.append({"label": resolved_classes[idx], "score": score})
                topk = sorted(topk, key=lambda x: x["score"], reverse=True)
                batch_results.append(ClassificationResult(topk[0]["label"], topk[0]["score"], topk))
                
        return batch_results

    def _export_dataset(self, output_path: str, format: str = "classification", **kwargs):
        """Classifies a directory of unlabeled images and saves them into class folders."""
        source = kwargs.get("source")
        if not source:
            raise ValueError("Parameter 'source' is required.")
            
        predict_kwargs = {k: v for k, v in kwargs.items() if k != "source"}
        images = media.load(source)
        verbose = kwargs.get("verbose", True)
        
        with ProgressBar(total=len(images), desc="Exporting Classification Dataset", verbose=verbose) as pbar:
            for idx, img in enumerate(images):
                res = self.predict(img, **predict_kwargs)
                class_dir = os.path.join(output_path, res.label)
                files.mkdir(class_dir)
                media.save(os.path.join(class_dir, f"classified_{idx}.jpg"), img)
                pbar.update(1)
