from .validators import InitModel, PredictModel, DetectModel

InitKernel = InitModel
PredictKernel = PredictModel
DetectKernel = DetectModel

__all__ = [
    "InitModel",
    "PredictModel",
    "DetectModel",
    "InitKernel",
    "PredictKernel",
    "DetectKernel",
]
