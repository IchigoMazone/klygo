"""Legacy models archive (LLM & VLM)."""
from .llm import LargeLanguageModel
from .vlm import VisionLanguageModel

Model = VisionLanguageModel

__all__ = ["LargeLanguageModel", "VisionLanguageModel", "Model"]

