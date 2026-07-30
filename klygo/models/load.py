import os
from typing import Any
from klygo import files
from klygo import config
from .metadata import MODEL_METADATA, REGISTRY_FILE_PATH
from .registry import registry
from .utils import decode_registry_key

from .base import FunctionalModelWrapper

_imported = False

def _ensure_imports():
    """Dynamically imports backends and adapters on first call to populate registry decorators."""
    global _imported
    if not _imported:
        import klygo.models.backends.torch
        import klygo.models.backends.huggingface
        import klygo.models.backends.ultralytics
        
        import klygo.models.adapters.detect
        import klygo.models.adapters.classify
        import klygo.models.adapters.segment
        import klygo.models.adapters.ocr
        import klygo.models.adapters.llm
        import klygo.models.adapters.speech
        import klygo.models.adapters.embedding
        _imported = True

def register(key: str = None, model_key: str = None, model_path: str = None, backend: str = None, task: str = None, loader: str = None) -> str:
    """
    Registers a model in the local persistent registry.
    
    Can be called either with:
    - A single 64-character encoded registry key (as the first positional arg).
    - Or explicit keyword arguments defining the model metadata.
    """
    _ensure_imports()
    
    if key is not None:
        meta = decode_registry_key(key)
        model_key = meta["model_key"]
        model_path = meta["model_path"]
        backend = meta["backend"]
        task = meta["task"]
        loader = meta.get("loader", None)
    else:
        if not (model_key and model_path and backend and task):
            raise ValueError("Explicit registration requires model_key, model_path, backend, and task parameters.")
            
    # Update in-memory registry
    MODEL_METADATA[model_key] = {
        "model_path": model_path,
        "backend": backend,
        "task": task
    }
    if loader:
        MODEL_METADATA[model_key]["loader"] = loader
        
    # Save to disk using config utility
    disk_registry = {}
    if files.exists(REGISTRY_FILE_PATH):
        try:
            disk_registry = dict(config.load(REGISTRY_FILE_PATH))
        except Exception:
            disk_registry = {}
            
    disk_registry[model_key] = MODEL_METADATA[model_key]
    config.save(REGISTRY_FILE_PATH, disk_registry, overwrite=True)
    
    return model_key

def load(key_or_path: Any, model_class: Any = None, task: str = None, **kwargs) -> Any:
    """
    Loads and instantiates a Klygo model.
    
    - If key_or_path matches a registered custom runner function: Wraps and returns it.
    - If model_class is provided: Loads a custom-coded PyTorch model from a .pth file.
    - If key_or_path is an instantiated PyTorch module: Wraps it directly in memory.
    - Otherwise: Loads a standard model from the registry using model_key.
    """
    _ensure_imports()

    # CASE 0: Custom functional runner registered via decorator
    if isinstance(key_or_path, str):
        model_func = registry.get_model_fn(key_or_path)
        if model_func is not None:
            return FunctionalModelWrapper(model_func, key_or_path)
    
    # CASE 1: Load custom PyTorch model directly via class/weights (bypass registry)
    if model_class is not None and key_or_path not in MODEL_METADATA:
        if not task:
            raise ValueError("Task parameter (e.g. task='classify') must be provided for custom model loading.")
            
        import torch
        raw_model = model_class()
        if key_or_path and files.exists(key_or_path):
            raw_model.load_state_dict(torch.load(key_or_path, map_location="cpu"))
            
        BackendClass = registry.get_backend("torch")
        AdapterClass = registry.get_adapter(task)
        
        backend_runner = BackendClass(raw_model)
        
        config_data = {
            "model_key": os.path.splitext(os.path.basename(key_or_path))[0] if isinstance(key_or_path, str) else "custom",
            "model_path": key_or_path,
            "backend": "torch",
            "task": task,
            "is_custom": True,
            **kwargs  # Contains classes/labels list
        }
        return AdapterClass(backend_runner, config_data)

    # CASE 2: Load instantiated PyTorch model object directly
    import torch
    if isinstance(key_or_path, torch.nn.Module):
        if not task:
            raise ValueError("Task parameter must be provided when loading an instantiated PyTorch module.")
            
        BackendClass = registry.get_backend("torch")
        AdapterClass = registry.get_adapter(task)
        backend_runner = BackendClass(key_or_path)
        
        config_data = {
            "model_key": "custom-in-memory",
            "backend": "torch",
            "task": task,
            "is_custom": True,
            **kwargs
        }
        return AdapterClass(backend_runner, config_data)

    # CASE 3: Standard registry model loading
    if key_or_path not in MODEL_METADATA:
        raise ValueError(f"Model '{key_or_path}' is not registered.")
        
    meta = MODEL_METADATA[key_or_path]
    BackendClass = registry.get_backend(meta["backend"])
    AdapterClass = registry.get_adapter(meta["task"])
    
    # Allow local model_path override for offline fallback if CDN is unreachable
    model_path = kwargs.pop("model_path", meta["model_path"])
    
    # Check if a custom loader function is registered
    custom_loader_name = meta.get("loader")
    if custom_loader_name:
        loader_fn = registry.get_loader_fn(custom_loader_name)
        if loader_fn is not None:
            raw_model = loader_fn(model_path)
            # Wrap raw model in backend
            backend_runner = BackendClass(raw_model)
        else:
            raise ValueError(f"Custom loader '{custom_loader_name}' is not registered.")
    else:
        # Load backend runner directly (load returns backend runner instance)
        backend_runner = BackendClass.load(model_path, model_class=model_class, **kwargs)
    
    config_data = {
        "model_key": key_or_path,
        "model_path": model_path,
        "backend": meta["backend"],
        "task": meta["task"],
        **kwargs
    }
    adapter = AdapterClass(backend_runner, config_data)
    return adapter

