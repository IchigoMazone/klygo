import sys
import os

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

def main():
    print("================================================================")
    print("Testing Model Recipe & Dynamic Parameter Validation System")
    print("================================================================\n")

    from klygo.models.registry import registry
    
    import torch
    
    class MockNet(torch.nn.Module):
        def __init__(self):
            super().__init__()

    @registry.register_loader_fn("my_mock_loader")
    def my_mock_loader(model_path):
        return MockNet()

    # 1. Register a model using an explicit template ID and mock loader
    print("Registering 'my-yolo' model with template 'yolov8n-detect'...")
    models.register(
        model_key="my-yolo",
        model_path="yolov8n.pt",
        backend="torch",
        task="detect",
        loader="my_mock_loader",
        template="yolov8n-detect"
    )

    # 2. Load the model
    print("\nLoading 'my-yolo' model...")
    model = models.load("my-yolo")
    
    # 3. View self-documenting help guide
    print("\nCalling model.help() to inspect guides, links, and parameters:")
    model.help()

    # 4. Test Parameter Validation & Default Injection
    print("TEST: Calling model.predict() with valid parameters:")
    try:
        # iou and verbose are omitted; they should resolve to default values 0.7 and False.
        # conf is passed explicitly.
        # Note: We override _predict to avoid torch backend trying to predict on mock model
        model._predict = lambda source, **kwargs: print(f"Mock predict executed with: {kwargs}")
        model.predict("dummy_image.jpg", conf=0.35)
        print("--> Success: Predict executed successfully.")
    except Exception as e:
        print(f"--> Error: {e}")

    # 5. Test Parameter Validation - Missing/Incorrect types
    print("\nTEST: Calling model.predict() with invalid parameter types (conf='high'):")
    try:
        model.predict("dummy_image.jpg", conf="high")
        print("--> Error: Parameter check failed (expected TypeError but it succeeded).")
    except TypeError as e:
        print(f"--> Success (Expected Error caught): {e}")

    # 6. Test Automatic Template Matching by Path
    print("\nRegistering a model without explicit template, matching by model_path name...")
    models.register(
        model_key="dino-auto-match",
        model_path="IDEA-Research/grounding-dino-tiny",
        backend="torch",
        task="detect",
        loader="my_mock_loader"
    )
    
    print("Loading 'dino-auto-match' (should auto-match 'grounding-dino-tiny' template)...")
    dino_model = models.load("dino-auto-match")
    dino_model.help()

    # 7. Test Case 1: Model has no pre-defined template/recipe at all (Dynamic Discovery)
    print("\n[Case 1]: Registering completely unknown model with no template...")
    
    class MysteriousNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def generate(self, input_ids, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 50):
            pass

    @registry.register_loader_fn("mysterious_loader")
    def mysterious_loader(model_path):
        return MysteriousNet()

    models.register(
        model_key="mysterious-model",
        model_path="unknown-author/mysterious-model-path",
        backend="torch",
        task="detect",
        loader="mysterious_loader"
    )

    print("Loading 'mysterious-model' (should trigger dynamic signature parameter discovery)...")
    mysterious_model = models.load("mysterious-model")
    mysterious_model.help()

if __name__ == "__main__":
    main()
