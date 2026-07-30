from typing import Any, List, Iterator
from .base import BaseAdapter
from ..registry import registry

class LLMResponse:
    def __init__(self, text: str, token_count: int = 0):
        self.text = text
        self.token_count = token_count

    def __repr__(self):
        return f"<LLMResponse: '{self.text[:30]}...'>"

@registry.register_adapter("llm")
class LLMAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> LLMResponse:
        """Alias for generate() to conform to BaseModel interface."""
        return self.generate(source, **kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        raw_out = self.backend.generate_text(prompt, **run_params)
        return LLMResponse(text=raw_out.get("text", ""), token_count=raw_out.get("tokens", 0))

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        raw_out = self.backend.generate_chat(messages, **run_params)
        return LLMResponse(text=raw_out.get("text", ""), token_count=raw_out.get("tokens", 0))

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        return self.backend.generate_stream(prompt, **run_params)
