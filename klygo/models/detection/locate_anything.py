from typing import Any, List, Dict, Union
import PIL.Image
from klygo.models import base
from klygo.outputs.detect import Detection


class LocateAnythingDetect(base.Detector):
    """
    Trình bao bọc mô hình nhận diện đối tượng kiến trúc LocateAnything (3B/Open-Vocabulary).
    """

    def __init__(self, metadata: Union[Dict[str, Any], str], **kwargs) -> None:
        super().__init__(metadata, **kwargs)

    def _infer_batch(
        self,
        batch_imgs: List[PIL.Image.Image],
        prompt: List[str],
        conf: float,
        text_threshold: float,
        half: bool,
    ) -> List[Detection]:
        raise NotImplementedError

    def help(self) -> None:
        """In ra thông tin mô hình và danh sách các hàm nghiệp vụ."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print(f"CLASS: {self.class_name}")
        print("=" * 60)
        print("1. predict(source, prompt, conf=0.25, batch=1, vid_stride=1, max_frames=None, half=False, device=None)")
        print("   Nhan dien doi tuong tren anh, video, folder tu media.load.")
        print("2. benchmark(data='data.yaml', iterations=20, warmup=5)")
        print("   Cham diem danh gia toc do suy luan (Do tre Latency ms / Toc do FPS).")
        print("3. export(output_dir='my_model')")
        print("   Xuat toan bo mo hinh (Weights + klygo.json) ra thu muc de chay Offline.")
