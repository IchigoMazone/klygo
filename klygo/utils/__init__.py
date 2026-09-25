"""General CUDA, formatting, and progress utilities (`klygo.utils`).

Public APIs (4 Functions, 1 Class):
    CUDA Inspection:
        1. is_cuda_available()             - Check whether CUDA is available.
        2. get_gpu_name(...)               - Return the selected GPU device name.

    Progress Reporting:
        3. ProgressBar(...)                - Context-managed progress indicator (class).
        4. create_progress_bar(...)        - Create a configured progress indicator.

    Formatting:
        5. human_size(num_bytes, ...)      - Format a byte count using binary units.
"""

from .cuda import is_cuda_available, get_gpu_name
from .formatting import human_size
from .progress import ProgressBar, create_progress_bar

__all__ = [
    "is_cuda_available",
    "get_gpu_name",
    "human_size",
    "ProgressBar",
    "create_progress_bar",
]
