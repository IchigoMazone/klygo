from klygo.cuda import is_available as is_cuda_available, get_device_name as get_gpu_name

__all__ = [
    "is_cuda_available",
    "get_gpu_name",
]
