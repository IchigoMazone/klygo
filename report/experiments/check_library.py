"""Record the existing Klygo test suite without hiding skips or failures."""
import contextlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "klygo"
DEST = SOURCE / "artifacts/report/logs"
DEST.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(DEST.parent / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(DEST.parent / "matplotlib"))
os.environ.setdefault("HF_HOME", str(DEST.parent / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_ids.append(test.id())


if __name__ == "__main__":
    begin = time.perf_counter()
    with (DEST / "unittest.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover("test", pattern="test*.py", top_level_dir=".")
            result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(suite)
    summary = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tests_run": result.testsRun, "passed": len(result.passed_ids),
        "failures": [{"test": test.id(), "traceback": reason} for test, reason in result.failures],
        "errors": [{"test": test.id(), "traceback": reason} for test, reason in result.errors],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
        "elapsed_seconds": time.perf_counter() - begin, "successful": result.wasSuccessful(),
        "passed_ids": result.passed_ids,
    }
    (DEST / "unittest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("passed_ids", "failures", "errors", "skipped")}, indent=2))
    print(f"Failures={len(result.failures)}, errors={len(result.errors)}, skipped={len(result.skipped)}")
    sys.exit(0 if result.wasSuccessful() else 1)
