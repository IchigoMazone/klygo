"""Launch the canonical Klygo release suite without duplicating its logic."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Forward every CLI argument from the repository root to evaluate_release."""
    # Bootstrap the checkout before Klygo itself is importable.
    root = Path(__file__).resolve().parents[4]
    runner = root / "benchmarks" / "gpu_suite" / "evaluate_release.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Canonical benchmark runner not found: {runner}")
    previous_cwd = Path.cwd()
    previous_path = sys.path[:]
    previous_argv = sys.argv[:]
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        sys.argv[0] = str(runner)
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.argv[:] = previous_argv


if __name__ == "__main__":
    main()
