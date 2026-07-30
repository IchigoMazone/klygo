from abc import ABC, abstractmethod
from typing import Any
from klygo import files

class BaseModel(ABC):
    def __init__(self, backend_runner: Any, config_data: dict):
        self.backend = backend_runner
        self.config = config_data
        
        # State tracking for training and prediction configs
        self._history = {
            "dataset": None,
            "operations": {
                "predict": {},
                "train": {},
                "evaluate": {}
            }
        }

    @property
    def native(self):
        """Returns the raw framework model instance (e.g. PyTorch module, HF model, YOLO)."""
        return self.backend.native if hasattr(self.backend, "native") else None

    def predict(self, *args, **kwargs) -> Any:
        model_key = self.config.get("model_key")
        from .registry import registry
        
        # Execute preprocess hook if registered
        preprocess_fn = registry.get_preprocess(model_key)
        if preprocess_fn is not None:
            processed = preprocess_fn(*args, **kwargs)
            if isinstance(processed, tuple):
                args = processed
            else:
                args = (processed,)
                
        # Dispatch to the adapter implementation
        result = self._predict(*args, **kwargs)
        
        # Execute postprocess hook if registered
        postprocess_fn = registry.get_postprocess(model_key)
        if postprocess_fn is not None:
            result = postprocess_fn(result, **kwargs)
            
        return result

    @abstractmethod
    def _predict(self, *args, **kwargs) -> Any:
        pass

    def train(self, *args, **kwargs) -> Any:
        # Implicitly capture dataset and training config parameters
        if "data" in kwargs:
            self._history["dataset"] = kwargs["data"]
        elif "dataset" in kwargs:
            self._history["dataset"] = kwargs["dataset"]
        elif len(args) > 0:
            self._history["dataset"] = args[0]
            
        self._history["operations"]["train"].update(kwargs)
        return self.backend.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Any:
        self._history["operations"]["evaluate"].update(kwargs)
        return self.backend.evaluate(*args, **kwargs)

    def save(self, path: str) -> None:
        self.backend.save(path)

    def unload(self) -> None:
        """Unloads model weights from GPU VRAM to system CPU memory to free up VRAM."""
        if hasattr(self.backend, "model") and hasattr(self.backend.model, "cpu"):
            try:
                self.backend.model.cpu()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Model '{self.config.get('model_key')}' unloaded from GPU VRAM.")
            except Exception as e:
                print(f"Warning: Failed to unload model: {e}")

    def warmup(self) -> None:
        """Runs a tiny dummy prediction to compile GPU kernels and initialize loaders upfront."""
        try:
            task = self.config.get("task", "")
            if task in ["detect", "classify", "segment", "ocr"]:
                from PIL import Image
                dummy_img = Image.new("RGB", (1, 1), color="black")
                self.predict(dummy_img, verbose=False)
            elif task == "llm":
                self.predict("warmup", max_tokens=1)
            elif task == "embedding":
                self.embed("warmup")
            elif task == "speech":
                pass
            print(f"Model '{self.config.get('model_key')}' warmed up successfully.")
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")

    # --- API 1: Export Key (64 chars) ---
    def export(self, output_path: str = None) -> str:
        """
        Generates a 64-character encoded registry key containing metadata.
        If output_path is provided, writes the key to a .txt file.
        """
        # Block export for custom in-memory models
        if self.config.get("is_custom", False):
            raise NotImplementedError(
                "In-memory custom-coded models cannot be exported to a registry key "
                "since the Python class definition is local to this machine."
            )
            
        model_path = self.config.get("model_path")
        backend = self.config.get("backend")
        task = self.config.get("task")
        model_key = self.config.get("model_key")
        
        from .utils import encode_registry_key
        key_64 = encode_registry_key(model_key, model_path, backend, task)
        
        if output_path:
            files.save(output_path, key_64, overwrite=True)
            
        return key_64

    # --- API 2: Save config to .yaml ---
    def save_config(self, output_path: str):
        """Saves the captured process configuration and operations to a .yaml file."""
        recipe = {
            "dataset": self._history["dataset"],
            "operations": self._history["operations"]
        }
        files.save(output_path, recipe, overwrite=True)

    # --- API 3: Load config from .yaml ---
    def load_config(self, input_path: str):
        """Loads and applies operations/dataset configuration from a .yaml file."""
        if not files.exists(input_path):
            raise FileNotFoundError(f"Config file not found: {input_path}")
            
        recipe = files.load(input_path)
        self._history["operations"] = recipe.get("operations", {})
        self._history["dataset"] = recipe.get("dataset", None)

    # --- API 4: Export Dataset (called by Detect/Segment adapters) ---
    def export_dataset(self, output_path: str, format: str, **kwargs) -> Any:
        """Dispatches dataset generation to internal method, implemented by CV adapters."""
        return self._export_dataset(output_path, format=format, **kwargs)

    def _export_dataset(self, output_path: str, format: str, **kwargs):
        raise NotImplementedError(f"Model task {self.__class__.__name__} does not support dataset exporting.")

    def info(self) -> dict:
        return {
            "backend": self.backend.__class__.__name__ if self.backend else "Functional",
            "task": self.__class__.__name__,
            "config": self.config,
            "history": self._history
        }

class FunctionalModelWrapper(BaseModel):
    def __init__(self, func: Any, model_key: str):
        super().__init__(None, {"model_key": model_key, "is_custom": True})
        self.func = func
        
    def _predict(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)
