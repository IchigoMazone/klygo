"""
Backend implementations for different framework engines (Hugging Face, Ultralytics).
"""

from . import huggingface
from . import huggingface as hf
from . import ultralytics

__all__ = ["huggingface", "hf", "ultralytics"]
