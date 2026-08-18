import os
from typing import Any
from klygo import files
from klygo import config
from .metadata import MODEL_METADATA, REGISTRY_FILE_PATH
from .registry import registry
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

from .recipe import recipe_manager

# Decentralized registry directory path
REGISTRY_DIR = os.path.expanduser("~/.klygo/registry")

def register(
    config_or_key: Any = None, 
    model_key: str = None, 
    model_path: str = None, 
    backend: str = None, 
    task: str = None, 
    loader: str = None, 
    predict_params: dict = None, 
    train_params: dict = None, 
    template: str = None
) -> str:
    """
    Registers a model in the decentralized registry.
    
    Can be called either with:
    - A path to a JSON configuration file (e.g. register("my_model.json")).
    - A dictionary containing the model metadata.
    - Explicit keyword arguments.
    """
    _ensure_imports()
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    
    meta = {}
    
    # Check if first positional arg is a JSON file path or a dict
    if config_or_key is not None:
        if isinstance(config_or_key, dict):
            meta = config_or_key
        elif isinstance(config_or_key, str):
            # Check if it points to a file on disk
            if files.exists(config_or_key) or config_or_key.endswith(".json"):
                try:
                    meta = config.load(config_or_key)
                except Exception as e:
                    raise ValueError(f"Failed to load JSON config file from {config_or_key}: {e}")
            else:
                raise ValueError("String parameter must be a valid path to a JSON metadata file.")
        else:
            raise TypeError("First parameter must be a JSON file path (str) or a metadata dictionary (dict).")
            
        # Extract metadata fields from file/dict
        model_key = meta.get("model_key") or model_key
        model_path = meta.get("model_path") or model_path
        backend = meta.get("backend") or backend
        task = meta.get("task") or task
        loader = meta.get("loader") or loader
        predict_params = meta.get("predict_params") or predict_params
        train_params = meta.get("train_params") or train_params
        template = meta.get("template") or template

    if not (model_key and model_path and backend and task):
        raise ValueError("Model registration requires model_key, model_path, backend, and task parameters.")
        
    # Build metadata dict
    model_data = {
        "model_key": model_key,
        "model_path": model_path,
        "backend": backend,
        "task": task
    }
    if loader:
        model_data["loader"] = loader
    if predict_params:
        model_data["predict_params"] = predict_params
    if train_params:
        model_data["train_params"] = train_params
    if template:
        model_data["template"] = template
        
    # Update in-memory registry
    MODEL_METADATA[model_key] = model_data
    
    # Save to individual decentralized JSON file
    model_config_path = os.path.join(REGISTRY_DIR, f"{model_key}.json")
    try:
        config.save(model_config_path, model_data, overwrite=True)
    except Exception as e:
        raise IOError(f"Failed to save decentralized model registry configuration to {model_config_path}: {e}")
        
    return model_key

def discover_params_from_backend(backend_runner: Any) -> dict:
    """Discovers supported predict parameters dynamically via signature reflection on Hugging Face components."""
    import inspect
    discovered = {}
    INTERNAL_PARAMS = {"self", "images", "text", "audio", "return_tensors", "args", "kwargs"}
    
    sigs = []
    # 1. Inspect processor
    processor = getattr(backend_runner, "processor", None)
    if processor and hasattr(processor, "__call__"):
        try:
            sigs.append(inspect.signature(processor.__call__))
        except Exception:
            pass
            
    # 2. Inspect tokenizer
    tokenizer = getattr(backend_runner, "tokenizer", None)
    if tokenizer and hasattr(tokenizer, "__call__"):
        try:
            sigs.append(inspect.signature(tokenizer.__call__))
        except Exception:
            pass
            
    # 3. Inspect model generate/forward
    model_obj = getattr(backend_runner, "model", None)
    if model_obj:
        if hasattr(model_obj, "generate"):
            try:
                sigs.append(inspect.signature(model_obj.generate))
            except Exception:
                pass
        elif hasattr(model_obj, "forward"):
            try:
                sigs.append(inspect.signature(model_obj.forward))
            except Exception:
                pass
                
    for sig in sigs:
        for name, param in sig.parameters.items():
            if name not in INTERNAL_PARAMS:
                type_name = "Any"
                if param.annotation is not inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        type_name = param.annotation.__name__
                    else:
                        type_name = str(param.annotation)
                discovered[name] = {
                    "type": type_name,
                    "required": param.default is inspect.Parameter.empty,
                    "default": param.default if param.default is not inspect.Parameter.empty else None,
                    "description": "Dynamically discovered HuggingFace parameter."
                }
    return discovered

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
    
    # Load template recipe if available
    template_id = meta.get("template")
    template_info = {}
    if template_id:
        template_info = recipe_manager.get_template(template_id)
    if not template_info:
        template_info = recipe_manager.match_template_by_path(model_path)
        
    # Merge parameter schemas
    merged_predict_params = {}
    merged_train_params = {}
    if template_info:
        merged_predict_params.update(template_info.get("predict_params", {}))
        merged_train_params.update(template_info.get("train_params", {}))
        
    merged_predict_params.update(meta.get("predict_params", {}))
    merged_train_params.update(meta.get("train_params", {}))
    
    # Fallback to Dynamic Parameter Discovery if schema is empty and backend is Hugging Face
    if not merged_predict_params and meta.get("backend") == "huggingface":
        merged_predict_params = discover_params_from_backend(backend_runner)
        
    config_data = {
        "model_key": key_or_path,
        "model_path": model_path,
        "backend": meta["backend"],
        "task": meta["task"],
        "predict_params": merged_predict_params,
        "train_params": merged_train_params,
        "guide": template_info.get("guide", "No guide available."),
        "links": template_info.get("links", {}),
        **kwargs
    }
    adapter = AdapterClass(backend_runner, config_data)
    return adapter

