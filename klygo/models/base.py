from abc import ABC, abstractmethod
from typing import Any
from klygo import files
import inspect

def bind_dynamic_ide_metadata(predict_method: Any, schema: dict, guide: str, links: dict):
    """
    Dynamically binds parameter signatures and docstrings to the predict method.
    This enables autocomplete and description popups directly in modern IDEs.
    """
    try:
        new_params = [
            inspect.Parameter("source", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any)
        ]
        type_mapping = {"str": str, "float": float, "int": int, "bool": bool}
        
        for param_name, info in schema.items():
            expected_type = type_mapping.get(info.get("type"), Any)
            if info.get("required"):
                default_val = inspect.Parameter.empty
            else:
                default_val = info.get("default", None)
                
            new_params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default_val,
                    annotation=expected_type
                )
            )
        predict_method.__func__.__signature__ = inspect.Signature(new_params)
        
        doc_lines = [
            f"[Guide]: {guide}",
            ""
        ]
        if links:
            doc_lines.append("[Documentation & Tutorials]:")
            for title, url in links.items():
                doc_lines.append(f"  * {title.replace('_', ' ').capitalize()}: {url}")
            doc_lines.append("")
            
        doc_lines.append("Parameters:")
        doc_lines.append("-----------")
        for param_name, info in schema.items():
            req = " (Required)" if info.get("required") else ""
            default = f", default: {info.get('default')}" if "default" in info else ""
            doc_lines.append(f"{param_name} : {info.get('type')}{req}{default}")
            doc_lines.append(f"    {info.get('description', 'No description.')}")
            
        predict_method.__func__.__doc__ = "\n".join(doc_lines)
    except Exception:
        pass

def generate_type_stubs(config_data: dict):
    """Generates local .pyi type stub files in the current workspace to enable static IDE autocomplete."""
    import os
    try:
        # Target directory in current working directory (workspace root)
        typings_dir = os.path.join(os.getcwd(), "typings", "klygo", "models")
        os.makedirs(typings_dir, exist_ok=True)
        
        # 1. Write typings/klygo/__init__.pyi
        klygo_init_path = os.path.join(os.getcwd(), "typings", "klygo", "__init__.pyi")
        with open(klygo_init_path, "w", encoding="utf-8") as f:
            f.write("# Type stubs for Klygo\n")
            
        # 2. Write typings/klygo/models/__init__.pyi
        models_init_path = os.path.join(typings_dir, "__init__.pyi")
        with open(models_init_path, "w", encoding="utf-8") as f:
            f.write("from .load import load, register, register_file\n")
            f.write("__all__ = ['load', 'register', 'register_file']\n")
            
        # 3. Write typings/klygo/models/load.pyi dynamically containing the custom signatures!
        load_pyi_path = os.path.join(typings_dir, "load.pyi")
        
        predict_params = config_data.get("predict_params", {})
        crop_params = config_data.get("crop_params", {})
        task = config_data.get("task", "")
        
        # Build predict signature string
        predict_args = ["self", "source: Any"]
        for name, info in predict_params.items():
            type_str = info.get("type", "Any")
            if info.get("required"):
                predict_args.append(f"{name}: {type_str}")
            else:
                predict_args.append(f"{name}: {type_str} = ...")
                
        predict_sig = ", ".join(predict_args)
        if len(predict_args) > 2:
            # Insert '*' to signify keyword-only arguments
            predict_args.insert(2, "*")
            predict_sig = ", ".join(predict_args)
            
        # Build crop signature string if it is a detect model
        crop_sig_lines = ""
        if task == "detect":
            crop_args = ["self", "source: Any"]
            for name, info in crop_params.items():
                type_str = info.get("type", "Any")
                if info.get("required"):
                    crop_args.append(f"{name}: {type_str}")
                else:
                    crop_args.append(f"{name}: {type_str} = ...")
            if len(crop_args) > 2:
                crop_args.insert(2, "*")
            crop_sig = ", ".join(crop_args)
            crop_sig_lines = f"    def crop({crop_sig}) -> Any: ...\n"
            
        # Build export_dataset signature
        export_dataset_lines = ""
        if task in ["detect", "classify"]:
            export_dataset_lines = "    def export_dataset(self, output_path: str, format: str, *, source: str = ..., classes: list = ..., **kwargs: Any) -> Any: ...\n"
            
        content = f"""from typing import Any

class LoadedModel:
    def predict({predict_sig}) -> Any: ...
{crop_sig_lines}{export_dataset_lines}    def unload(self) -> None: ...
    def warmup(self) -> None: ...
    def help(self) -> None: ...
    def export_config(self, output_path: str) -> None: ...

def load(key_or_path: str, **kwargs: Any) -> LoadedModel: ...
def register(model_key: str, model_path: str, backend: str, task: str, **kwargs: Any) -> str: ...
def register_file(file_path: str) -> str: ...
"""

        with open(load_pyi_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass

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
        
        # Bind signature and help docstring dynamically
        if hasattr(self, "predict") and hasattr(self.predict, "__func__"):
            bind_dynamic_ide_metadata(
                predict_method=self.predict,
                schema=self.get_predict_params(),
                guide=self.config.get("guide", "No guide available."),
                links=self.config.get("links", {})
            )
        
        # Generate static type stubs for local workspace
        generate_type_stubs(self.config)

    @property
    def native(self):
        """Returns the raw framework model instance (e.g. PyTorch module, HF model, YOLO)."""
        return self.backend.native if hasattr(self.backend, "native") else None

    def predict(self, *args, **kwargs) -> Any:
        model_key = self.config.get("model_key")
        from .registry import registry
        
        # Validate parameters and inject defaults based on predict_params schema
        predict_params = self.get_predict_params()
        if predict_params:
            for param_name, info in predict_params.items():
                if info.get("required") and param_name not in kwargs:
                    raise ValueError(f"Parameter '{param_name}' is required for model '{model_key}' but was not provided.")
                if param_name not in kwargs and "default" in info:
                    kwargs[param_name] = info["default"]
                elif param_name in kwargs:
                    expected_type_str = info.get("type", "Any")
                    val = kwargs[param_name]
                    type_mapping = {"str": str, "float": float, "int": int, "bool": bool}
                    expected_type = type_mapping.get(expected_type_str)
                    if expected_type and not isinstance(val, expected_type):
                        try:
                            # Try to safely cast
                            kwargs[param_name] = expected_type(val)
                        except Exception:
                            raise TypeError(f"Parameter '{param_name}' must be of type {expected_type_str}, got {type(val).__name__} instead.")
        
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

    # --- API 1: Export Configuration ---
    def export(self, output_path: str):
        """Exports the model configuration/metadata to a JSON file for community sharing."""
        import json
        try:
            export_data = {
                "model_key": self.config.get("model_key"),
                "model_path": self.config.get("model_path"),
                "backend": self.config.get("backend"),
                "task": self.config.get("task"),
                "loader": self.config.get("loader"),
                "template": self.config.get("template"),
                "predict_params": self.config.get("predict_params", {}),
                "train_params": self.config.get("train_params", {}),
                "guide": self.config.get("guide"),
                "links": self.config.get("links", {})
            }
            # Remove None values to keep JSON clean
            export_data = {k: v for k, v in export_data.items() if v is not None}
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            print(f"-> Exported model configuration to: {output_path}")
        except Exception as e:
            print(f"Error: Failed to export metadata JSON: {e}")

    def info(self) -> dict:
        return {
            "backend": self.backend.__class__.__name__ if self.backend else "Functional",
            "task": self.__class__.__name__,
            "config": self.config,
            "history": self._history
        }

    def get_predict_params(self) -> dict:
        """Returns the registered standard prediction parameters schema for this model."""
        return self.config.get("predict_params", {})

    def get_train_params(self) -> dict:
        """Returns the registered standard training parameters schema for this model."""
        return self.config.get("train_params", {})

    def help(self):
        """Prints a beautifully formatted user-friendly guide of the model parameters."""
        print(f"============================================================")
        print(f"MODEL: {self.config.get('model_key')} ({self.config.get('backend')}/{self.config.get('task')})")
        print(f"============================================================")
        
        # Display guide and links if available
        guide = self.config.get("guide")
        if guide:
            print(f"[Guide]: {guide}")
            
        links = self.config.get("links", {})
        if links:
            print("\n[Documentation & Example Tutorials]:")
            for title, url in links.items():
                print(f"  * {title.replace('_', ' ').capitalize()}: {url}")
                
        predict_params = self.get_predict_params()
        print("\n[Predict Parameters (predict())]:")
        if not predict_params:
            print("  No parameters registered.")
        else:
            for name, info in predict_params.items():
                req = " (Required)" if info.get("required") else ""
                default = f", Default: {info.get('default')}" if "default" in info else ""
                print(f"  * {name} ({info.get('type', 'Any')}){req}{default}")
                print(f"    Description: {info.get('description', 'No description.')}")
                
        train_params = self.get_train_params()
        print("\n[Train Parameters (train())]:")
        if not train_params:
            print("  No parameters registered.")
        else:
            for name, info in train_params.items():
                req = " (Required)" if info.get("required") else ""
                default = f", Default: {info.get('default')}" if "default" in info else ""
                print(f"  * {name} ({info.get('type', 'Any')}){req}{default}")
                print(f"    Description: {info.get('description', 'No description.')}")
    def update_predict_params(self, **updated_schema):
        """Updates model parameter definitions and saves them to the individual JSON configuration file."""
        import os
        import json
        
        registry_dir = os.path.expanduser("~/.klygo/registry")
        os.makedirs(registry_dir, exist_ok=True)
        model_key = self.config.get("model_key")
        
        # Merge changes in self.config
        if "predict_params" not in self.config:
            self.config["predict_params"] = {}
        self.config["predict_params"].update(updated_schema)
        
        # Save to individual JSON file
        filepath = os.path.join(registry_dir, f"{model_key}.json")
        try:
            export_data = {
                "model_key": self.config.get("model_key"),
                "model_path": self.config.get("model_path"),
                "backend": self.config.get("backend"),
                "task": self.config.get("task"),
                "loader": self.config.get("loader"),
                "template": self.config.get("template"),
                "predict_params": self.config.get("predict_params", {}),
                "train_params": self.config.get("train_params", {}),
                "guide": self.config.get("guide"),
                "links": self.config.get("links", {})
            }
            # Remove keys with None values to keep JSON clean
            export_data = {k: v for k, v in export_data.items() if v is not None}
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            print(f"Model parameters successfully updated and saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save updated configuration file to disk: {e}")
            
        # Re-bind signature and help docstring dynamically
        bind_dynamic_ide_metadata(
            predict_method=self.predict,
            schema=self.get_predict_params(),
            guide=self.config.get("guide", "No guide available."),
            links=self.config.get("links", {})
        )
        
        # Regenerate static type stubs for local workspace
        generate_type_stubs(self.config)



class FunctionalModelWrapper(BaseModel):
    def __init__(self, func: Any, model_key: str):
        super().__init__(None, {"model_key": model_key, "is_custom": True})
        self.func = func
        
    def _predict(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)
