import os
import json
import urllib.request
import threading
from klygo import files

RECIPES_CACHE_PATH = os.path.expanduser("~/.klygo_recipes.json")

# Predefined templates built into the library (Offline fallbacks)
DEFAULT_RECIPES = {
    "yolov8n-detect": {
        "name": "YOLOv8n Object Detection",
        "guide": "High-speed real-time 2D object detection model by Ultralytics.",
        "links": {
            "tutorial_code": "https://github.com/klygo/examples/blob/main/yolov8_tutorial.py",
            "docs": "https://docs.ultralytics.com"
        },
        "predict_params": {
            "conf": {"type": "float", "default": 0.25, "description": "Object confidence threshold"},
            "iou": {"type": "float", "default": 0.7, "description": "IoU threshold for non-maximum suppression"},
            "verbose": {"type": "bool", "default": False, "description": "Enable verbose print logs"}
        }
    },
    "grounding-dino-tiny": {
        "name": "Grounding DINO Tiny",
        "guide": "Zero-shot visual grounding detector for locating arbitrary objects via text prompts.",
        "links": {
            "tutorial_code": "https://github.com/klygo/examples/blob/main/grounding_dino_tutorial.py",
            "docs": "https://huggingface.co/IDEA-Research/grounding-dino-tiny"
        },
        "predict_params": {
            "text_prompt": {"type": "str", "required": True, "description": "Visual query containing search phrases separated by dots"},
            "box_threshold": {"type": "float", "default": 0.3, "description": "Threshold for bounding box detection"},
            "text_threshold": {"type": "float", "default": 0.25, "description": "Text similarity threshold"}
        }
    },
    "nvidia-locate-anything-3b": {
        "name": "NVIDIA LocateAnything-3B",
        "guide": "State-of-the-art vision-language visual grounding model with Parallel Box Decoding.",
        "links": {
            "tutorial_code": "https://github.com/klygo/examples/blob/main/locate_anything_tutorial.py",
            "docs": "https://huggingface.co/nvidia/LocateAnything-3B"
        },
        "predict_params": {
            "text_prompt": {"type": "str", "required": True, "description": "Prompt specifying objects to detect"},
            "box_threshold": {"type": "float", "default": 0.3, "description": "Confidence threshold"},
            "text_threshold": {"type": "float", "default": 0.25, "description": "Association threshold"}
        }
    }
}

class RecipeManager:
    def __init__(self):
        self.recipes = dict(DEFAULT_RECIPES)
        self.load_cache()
        self.fetch_updates_async()

    def load_cache(self):
        if files.exists(RECIPES_CACHE_PATH):
            try:
                cached = files.load(RECIPES_CACHE_PATH)
                if isinstance(cached, dict):
                    self.recipes.update(cached)
            except Exception:
                pass

    def fetch_updates_async(self):
        thread = threading.Thread(target=self._fetch_updates, daemon=True)
        thread.start()

    def _fetch_updates(self):
        # Public CDN repository for dynamic recipe lists
        url = "https://raw.githubusercontent.com/klygo/recipes/main/recipes.json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                remote_recipes = json.loads(response.read().decode('utf-8'))
                if isinstance(remote_recipes, dict):
                    self.recipes.update(remote_recipes)
                    # Cache to disk
                    files.save(RECIPES_CACHE_PATH, remote_recipes, overwrite=True)
        except Exception:
            # Connection failed or timed out (offline mode fallback)
            pass

    def get_template(self, template_id: str) -> dict:
        return self.recipes.get(template_id, {})

    def match_template_by_path(self, model_path: str) -> dict:
        """Finds matching recipe based on model path keyword match."""
        path_lower = model_path.lower()
        if "yolov8" in path_lower:
            return self.get_template("yolov8n-detect")
        elif "grounding-dino" in path_lower:
            return self.get_template("grounding-dino-tiny")
        elif "locateanything" in path_lower:
            return self.get_template("nvidia-locate-anything-3b")
        return {}

recipe_manager = RecipeManager()
