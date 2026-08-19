from typing import Any, List
from ..interfaces import DetectorModel, DetectionResult, CropResult


class YOLODetect(DetectorModel):
    """
    Trình bao bọc mô hình nhận diện đối tượng kiến trúc YOLO.
    """

    def __init__(self, task: str, backend: str, num_params: str, **kwargs) -> None:
        self.task = task
        self.backend = backend
        self.num_params = num_params
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    def to(self, device_name: str) -> "YOLODetect":
        self._device = device_name
        return self

    def predict(
        self,
        source: Any,
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        raise NotImplementedError

    def crop(
        self,
        source: Any,
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> CropResult:
        raise NotImplementedError

    def export(self, output_path: str, format: str = "onnx", half: bool = False) -> str:
        raise NotImplementedError

    def dataset(
        self,
        output_path: str,
        format: str,
        source: str,
        text_prompt: List[str],
        batch_size: int = 16,
        threshold: float = 0.4,
    ) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def help(self) -> None:
        print(f"MODEL: YOLO ({self.backend}/{self.task})")
        print("=" * 52)
        print("1. predict(source: Any, text_prompt: List[str], threshold=0.4, text_threshold=0.3)")
        print("2. crop(source: Any, text_prompt: List[str], threshold=0.4, text_threshold=0.3)")
        print("3. dataset(output_path: str, format: str, source: str, text_prompt: List[str], batch_size=16, threshold=0.4)")
        print("4. export(output_path: str, format='onnx', half=False)")
