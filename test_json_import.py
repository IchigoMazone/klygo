import sys
import os
import json

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

def main():
    print("================================================================")
    print("Testing Decentralized JSON-Import Model Registration")
    print("================================================================\n")

    # 1. Create a sample model metadata JSON file
    test_json_path = "test_yolo_metadata.json"
    metadata = {
        "model_key": "imported-yolo",
        "model_path": "yolov8n.pt",
        "backend": "torch",
        "task": "detect",
        "loader": "my_mock_loader",
        "template": "yolov8n-detect",
        "predict_params": {
            "conf": {
                "type": "float",
                "default": 0.35,
                "description": "User-optimized object detection threshold"
            }
        }
    }
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Created sample metadata JSON file: {test_json_path}")

    # 2. Mock a custom loader to bypass weights loading
    from klygo.models.registry import registry
    import torch
    
    class MockNet(torch.nn.Module):
        def __init__(self):
            super().__init__()

    @registry.register_loader_fn("my_mock_loader")
    def my_mock_loader(model_path):
        return MockNet()

    # 3. Register model by passing the JSON file path directly!
    print("\nRegistering model via JSON file path import...")
    models.register(test_json_path)
    print("Registration completed.")

    # 4. Load the model and verify the imported configuration is active
    print("\nLoading the registered model 'imported-yolo'...")
    model = models.load("imported-yolo")
    
    # 5. Display help to verify custom conf description & default (0.35 instead of 0.25)
    model.help()

    # 6. Clean up temporary files
    if os.path.exists(test_json_path):
        os.remove(test_json_path)
        print(f"Removed temporary JSON file {test_json_path}")

if __name__ == "__main__":
    main()
