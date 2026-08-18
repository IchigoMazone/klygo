import os
from klygo import files
from klygo import config

REGISTRY_FILE_PATH = os.path.expanduser("~/.klygo_registry.json")

MODEL_METADATA = {
    "yolov8n-detect": {
        "model_path": "yolov8n.pt",
        "backend": "ultralytics",
        "task": "detect",
        "predict_params": {
            "conf": {"type": "float", "default": 0.25, "description": "Object confidence threshold for detection"},
            "iou": {"type": "float", "default": 0.7, "description": "Intersection over Union (IoU) threshold for NMS"},
            "verbose": {"type": "bool", "default": False, "description": "Whether to print inference details"}
        }
    },
    "grounding-dino-tiny": {
        "model_path": "IDEA-Research/grounding-dino-tiny",
        "backend": "huggingface",
        "task": "detect",
        "predict_params": {
            "text_prompt": {"type": "str", "required": True, "description": "Classes to search for separated by dots (e.g., 'cat . dog')"},
            "box_threshold": {"type": "float", "default": 0.3, "description": "Bounding box threshold"},
            "text_threshold": {"type": "float", "default": 0.25, "description": "Text similarity threshold"}
        }
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
