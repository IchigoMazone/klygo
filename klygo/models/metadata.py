import os
from klygo import files
from klygo import config

REGISTRY_FILE_PATH = os.path.expanduser("~/.klygo_registry.json")

MODEL_METADATA = {
    "yolov8n-detect": {
        "model_path": "yolov8n.pt",
        "backend": "ultralytics",
        "task": "detect"
    },
    "grounding-dino-tiny": {
        "model_path": "IDEA-Research/grounding-dino-tiny",
        "backend": "huggingface",
        "task": "detect"
    }
}

def load_persistent_registry():
    """Loads previously registered models from the user config directory on start."""
    if files.exists(REGISTRY_FILE_PATH):
        try:
            persisted_models = config.load(REGISTRY_FILE_PATH)
            MODEL_METADATA.update(persisted_models)
        except Exception as e:
            print(f"Warning: Failed to load persistent registry from disk: {e}")

# Load persistent models on module load
load_persistent_registry()
