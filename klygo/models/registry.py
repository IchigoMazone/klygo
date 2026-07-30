from typing import Callable, Any

class ModelRegistry:
    def __init__(self):
        self._backends = {}
        self._adapters = {}
        
        # Custom Function Registration Maps
        self._model_funcs = {}
        self._loader_funcs = {}
        self._preprocess_hooks = {}
        self._postprocess_hooks = {}

    def register_backend(self, name: str):
        def decorator(cls):
            self._backends[name] = cls
            return cls
        return decorator

    def register_adapter(self, name: str):
        def decorator(cls):
            self._adapters[name] = cls
            return cls
        return decorator

    def register_model_fn(self, name: str):
        """Decorator to register a custom runner function directly as a model."""
        def decorator(func: Callable[..., Any]):
            self._model_funcs[name] = func
            return func
        return decorator

    def register_loader_fn(self, name: str):
        """Decorator to register a custom loading function for weights loading."""
        def decorator(func: Callable[..., Any]):
            self._loader_funcs[name] = func
            return func
        return decorator

    def register_preprocess(self, model_key: str):
        """Decorator to hook a custom preprocessing function to a model."""
        def decorator(func: Callable[..., Any]):
            self._preprocess_hooks[model_key] = func
            return func
        return decorator

    def register_postprocess(self, model_key: str):
        """Decorator to hook a custom postprocessing function to a model."""
        def decorator(func: Callable[..., Any]):
            self._postprocess_hooks[model_key] = func
            return func
        return decorator

    def get_backend(self, name: str):
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' is not registered.")
        return self._backends[name]

    def get_adapter(self, name: str):
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' is not registered.")
        return self._adapters[name]

    def get_model_fn(self, name: str):
        return self._model_funcs.get(name)

    def get_loader_fn(self, name: str):
        return self._loader_funcs.get(name)

    def get_preprocess(self, model_key: str):
        return self._preprocess_hooks.get(model_key)

    def get_postprocess(self, model_key: str):
        return self._postprocess_hooks.get(model_key)

registry = ModelRegistry()
