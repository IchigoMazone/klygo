from .vlm import VisionLanguageModel
from .llm import LargeLanguageModel
from typing import Any

def load_model() -> Any:
    pass

__all__ = [
    "VisionLanguageModel",
    "LargeLanguageModel",
    "load_model"
]
