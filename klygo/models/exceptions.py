class KlygoModelError(Exception):
    """Base exception for all Klygo Models errors."""
    pass

class ModelLoadError(KlygoModelError):
    """Raised when a model fails to load or weights are missing/invalid."""
    pass

class BackendExecutionError(KlygoModelError):
    """Raised during model execution (predict/train/evaluate) in the framework backend."""
    pass
