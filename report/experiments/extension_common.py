"""Pinned extension revision, independent from the original report environment."""
from pathlib import Path
import os
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
REVISION = "c4388f10bba7cf5ff02b9ebc4aa217b9f740e759"
SOURCE = ROOT / "tmp/klygo-c4388f1"
OUT = ROOT / "klygo/artifacts/report/extension"
DATA = ROOT / "klygo/artifacts/report/data"
WEIGHTS = ROOT / "klygo/artifacts/report/weights"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))
os.environ["YOLO_CONFIG_DIR"] = str(OUT / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib")

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
