from typing import Any, Union
from .base import BaseAdapter
from ..registry import registry
from klygo import media
from klygo import files
import numpy as np

class EmbeddingResult:
    def __init__(self, vector: np.ndarray):
        self.vector = vector

    def __repr__(self):
        return f"<EmbeddingResult: shape={self.vector.shape}>"

@registry.register_adapter("embedding")
class EmbeddingAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> EmbeddingResult:
        """Alias for embed() to conform to BaseModel interface."""
        vector = self.embed(source, **kwargs)
        return EmbeddingResult(vector)

    def embed(self, source: Union[str, Any], **kwargs) -> np.ndarray:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        # Check if text input or image path
        if isinstance(source, str) and not files.exists(source):
            vector = self.backend.embed_text(source, **run_params)
        else:
            image = media.load(source)[0]
            vector = self.backend.embed_image(image, **run_params)
            
        return np.array(vector)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates cosine similarity between two embedding vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
