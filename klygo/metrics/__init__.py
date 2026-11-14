"""Composable, backend-neutral object-detection evaluation."""
from .operations import *
from .operations import __all__ as _operations
from .evaluation import compose, evaluate

__all__ = [*_operations, "compose", "evaluate"]
