"""Object detection wrapper for LocateAnything architectures (`klygo.models.detection.locate_anything`).

.. note::

    This class is a registered placeholder.  A concrete implementation will be
    added when official LocateAnything inference weights and an HF-compatible
    processor become publicly available.
"""

from typing import Dict, Any, List, Union
import PIL.Image

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection

__all__ = ["LocateAnythingDetect"]


class LocateAnythingDetect(Detector):
    """Placeholder detector for LocateAnything models.

    Raises :class:`NotImplementedError` on every forward call until a concrete
    inference backend is available.
    """

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]] = None,
    ) -> List[Detection]:
        """Not yet implemented — raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "LocateAnythingDetect.forward() is not yet implemented. "
            "A concrete inference backend for LocateAnything models has not been integrated. "
            "Use GroundingDinoDetect or YoloWorldDetect for zero-shot detection."
        )
