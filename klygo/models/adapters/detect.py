import os
from typing import Any, List
from .base import BaseAdapter
from ..registry import registry
from klygo import media
from klygo import files
from klygo.utils.progress import ProgressBar

class BoundingBox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, label: str, score: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.score = score

    def __repr__(self):
        return f"[{self.label}: {self.score:.2f} -> ({int(self.xmin)}, {int(self.ymin)}, {int(self.xmax)}, {int(self.ymax)})]"

class DetectionResult:
    def __init__(self, objects: List[BoundingBox]):
        self.objects = objects

    def __repr__(self):
        return f"<DetectionResult: found {len(self.objects)} objects>"

@registry.register_adapter("detect")
class DetectAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> Any:
        # Load default parameters from history config
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        
        # Capture parameters in history
        self._history["operations"]["predict"].update(kwargs)
        
        from PIL import Image
        import numpy as np
        
        # Enforce single image input (Image object, numpy array, or path to single image file)
        if isinstance(source, (Image.Image, np.ndarray)):
            images = [source]
        elif isinstance(source, str) and not os.path.isdir(source) and not source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            images = media.load(source)
        else:
            raise TypeError(
                "source must be a single PIL Image, numpy array, or a filepath to a single image file. "
                "Folders and videos must be processed frame-by-frame using media.load()."
            )
            
        raw_outputs = self.backend.predict(images, **run_params)
        results = self._post_process(raw_outputs, images)
        return results[0]

    def crop(self, source: Any, **kwargs) -> List[Any]:
        """Runs object detection and crops each detected object into a separate PIL Image."""
        from PIL import Image
        import numpy as np
        
        # Enforce single image input
        if isinstance(source, (Image.Image, np.ndarray)):
            img = source
        elif isinstance(source, str) and not os.path.isdir(source) and not source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            img = media.load(source)[0]
        else:
            raise TypeError(
                "source must be a single PIL Image, numpy array, or a filepath to a single image file. "
                "Folders and videos must be processed frame-by-frame using media.load()."
            )
            
        results = self.predict(img, **kwargs)
        
        cropped_results = []
        for obj in results.objects:
            # Clip box coordinates to image size boundaries
            xmin = max(0, int(obj.xmin))
            ymin = max(0, int(obj.ymin))
            xmax = min(img.width, int(obj.xmax))
            ymax = min(img.height, int(obj.ymax))
            
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            cropped_results.append(cropped_img)
            
        return cropped_results

    def export_dataset(self, output_path: str, format: str, **kwargs) -> Any:
        """Generates YOLO or cropped Classification dataset folder structure from predictions."""
        from klygo.datasets import detect
        source = kwargs.get("source")
        classes = kwargs.get("classes")
        return detect.export(
            model=self,
            output_path=output_path,
            format=format,
            source=source,
            classes=classes,
            **kwargs
        )

    def _post_process(self, raw_outputs: Any, images: list) -> List[DetectionResult]:
        if raw_outputs is None:
            return [DetectionResult([]) for _ in images]

        batch_results = []

        # Case A: Hugging Face zero-shot / Grounding DINO outputs
        if isinstance(raw_outputs, list) and isinstance(raw_outputs[0], dict) and "boxes" in raw_outputs[0]:
            for output in raw_outputs:
                objects = []
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    objects.append(BoundingBox(
                        xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3],
                        label=label, score=score
                    ))
                batch_results.append(DetectionResult(objects))

        # Case B: Ultralytics YOLO outputs
        elif isinstance(raw_outputs, list) and hasattr(raw_outputs[0], "boxes"):
            for result in raw_outputs:
                objects = []
                names = result.names
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = names[class_id]
                    objects.append(BoundingBox(
                        xmin=coords[0], ymin=coords[1], xmax=coords[2], ymax=coords[3],
                        label=label, score=score
                    ))
                batch_results.append(DetectionResult(objects))
        else:
            batch_results = [DetectionResult([]) for _ in images]

        return batch_results

    def update_crop_params(self, **updated_schema):
        """Updates crop parameter definitions and saves them to the individual JSON configuration file."""
        if "crop_params" not in self.config:
            self.config["crop_params"] = {}
        self.config["crop_params"].update(updated_schema)
        
        # Save config file using standard BaseModel logic
        import os
        import json
        registry_dir = os.path.expanduser("~/.klygo/registry")
        filepath = os.path.join(registry_dir, f"{self.config.get('model_key')}.json")
        try:
            # Clean export keys
            export_data = {
                "model_key": self.config.get("model_key"),
                "model_path": self.config.get("model_path"),
                "backend": self.config.get("backend"),
                "task": self.config.get("task"),
                "loader": self.config.get("loader"),
                "template": self.config.get("template"),
                "predict_params": self.config.get("predict_params", {}),
                "crop_params": self.config.get("crop_params", {}),
                "train_params": self.config.get("train_params", {}),
                "guide": self.config.get("guide"),
                "links": self.config.get("links", {})
            }
            export_data = {k: v for k, v in export_data.items() if v is not None}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            print(f"Crop parameters successfully updated and saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save updated configuration file to disk: {e}")
            
        # Re-generate type stubs dynamically!
        from klygo.models.base import generate_type_stubs
        generate_type_stubs(self.config)


