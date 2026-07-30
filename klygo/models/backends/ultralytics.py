from typing import Any
from ..registry import registry
from ..exceptions import ModelLoadError, BackendExecutionError

@registry.register_backend("ultralytics")
class UltralyticsBackend:
    def __init__(self, raw_model, device: str = "cpu", **kwargs):
        self.model = raw_model
        self.device = device

    @classmethod
    def load(cls, model_path: str, device: str = "cpu", **kwargs):
        """Loads Ultralytics YOLO model from file using lazy imports."""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return cls(model, device=device, **kwargs)
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from {model_path}: {e}")

    @property
    def native(self):
        return self.model

    def predict(self, images: list, **kwargs) -> list:
        """Executes YOLO predict."""
        # confidence threshold defaults to conf
        conf = kwargs.get("confidence", kwargs.get("box_threshold", 0.25))
        results = self.model.predict(
            source=images,
            device=self.device,
            conf=conf,
            verbose=False
        )
        return results

    def train(self, *args, **kwargs) -> Any:
        """Runs the native Ultralytics training loop."""
        return self.model.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> dict:
        """Runs validation metrics evaluation on YOLO model."""
        # val() is native validator
        metrics = self.model.val(*args, **kwargs)
        return metrics.results_dict

    def save(self, path: str):
        """Saves weights (standard file copying for YOLO, or native save)."""
        try:
            self.model.save(path)
        except Exception as e:
            raise BackendExecutionError(f"Failed to save YOLO weights to {path}: {e}")
