"""
Backend implementations for different framework engines (Hugging Face, Ultralytics).
"""

from . import hf
from . import ultralytics

__all__ = ["hf", "ultralytics"]
