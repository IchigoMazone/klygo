"""Exceptions raised by model configuration and runtime operations."""

__all__ = [
    "ModelFrameworkError", "InvalidStateError", "InvalidConfigError",
    "InvalidDeviceError", "InvalidDtypeError",
]


class ModelFrameworkError(Exception):
    """Base exception for all Klygo AI model configuration and runtime errors."""
    pass


class InvalidStateError(ModelFrameworkError):
    """Raised when a model operation is invoked in an invalid lifecycle state (e.g., calling ``predict()`` after ``unload()``)."""
    pass


class InvalidConfigError(ModelFrameworkError, ValueError):
    """Raised when model metadata or configuration parameters are invalid."""
    pass


class InvalidDeviceError(ModelFrameworkError, ValueError):
    """Raised when a requested hardware accelerator or device specification is invalid."""
    pass


class InvalidDtypeError(ModelFrameworkError, ValueError):
    """Raised when a requested numerical precision (dtype) is unsupported by the active backend."""
    pass
