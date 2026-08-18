import sys
import os

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

def main():
    print("====================================================================")
    print("Klygo Models Self-Documenting Parameters Schema Verification")
    print("====================================================================\n")

    # 1. Register a model with custom parameter schemas
    models.register(
        model_key="nvidia-locate-anything",
        model_path="nvidia/LocateAnything-3B",
        backend="huggingface",
        task="detect",
        predict_params={
            "text_prompt": {"type": "str", "required": True, "description": "Visual grounding prompt class queries"},
            "box_threshold": {"type": "float", "default": 0.3, "description": "Minimum confidence for bounding box filtering"},
            "text_threshold": {"type": "float", "default": 0.25, "description": "Text alignment similarity threshold"}
        }
    )

    # 2. Load the model but bypass actual framework download/load using a mock model class
    # (since we just want to verify metadata schema resolution in models.load and BaseModel)
    class DummyNet:
        pass
    
    # We will instantiate BaseModel with mock configurations to test helper methods
    # Wait, we can test it directly on a registered model that we load as a mock PyTorch model,
    # or by wrapping a dummy network directly!
    # But wait, case 3 standard load instantiates actual backend, which would fail if we don't have GPU/weights.
    # To test without loading weights, we can instantiate BaseModel directly:
    print("Creating mock BaseModel instances to display parameters help schemas:\n")
    
    # YOLO Mock
    yolo_model = models.load("yolov8n-detect", model_class=DummyNet, task="detect")
    yolo_model.help()
    
    print("\n")
    
    # LocateAnything Mock
    nvidia_model = models.load("nvidia-locate-anything", model_class=DummyNet, task="detect")
    nvidia_model.help()

if __name__ == "__main__":
    main()
