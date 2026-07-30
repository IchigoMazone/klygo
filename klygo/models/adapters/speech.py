from typing import Any
from .base import BaseAdapter
from ..registry import registry

class SpeechTranscript:
    def __init__(self, text: str, chunks: list = None):
        self.text = text
        self.chunks = chunks or []

    def __repr__(self):
        return f"<SpeechTranscript: '{self.text[:30]}...'>"

@registry.register_adapter("speech")
class SpeechAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> SpeechTranscript:
        """Alias for transcribe() to conform to BaseModel interface."""
        return self.transcribe(source, **kwargs)

    def transcribe(self, audio_path: str, **kwargs) -> SpeechTranscript:
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        self._history["operations"]["predict"].update(kwargs)
        
        raw_out = self.backend.transcribe(audio_path, **run_params)
        return SpeechTranscript(text=raw_out.get("text", ""), chunks=raw_out.get("chunks", []))
