import sys
import os

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

def main():
    print("====================================================================")
    print("Verifying Complete Developer Lifecycle: Register -> Load -> Update -> Export")
    print("====================================================================\n")

    # 1. Mock custom loader for testing offline
    from klygo.models.registry import registry
    import torch
    
    class MockNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def generate(self, input_ids, max_new_tokens: int = 150, temperature: float = 0.8):
            pass

    @registry.register_loader_fn("my_mock_loader")
    def my_mock_loader(model_path):
        return MockNet()

    # Step 1: Register a new model with basic metadata (no parameter schema)
    print("Step 1: Registering 'my-new-vlm' with loader...")
    models.register(
        model_key="my-new-vlm",
        model_path="facebook/brand-new-vlm",
        backend="huggingface",
        task="detect",
        loader="my_mock_loader"
    )

    # Step 2: Load the model (triggers dynamic signature discovery)
    print("\nStep 2: Loading 'my-new-vlm'...")
    model = models.load("my-new-vlm")
    
    print("\nVerifying automatically discovered draft signature:")
    model.help()

    # Step 3: Developer updates parameters and advanced descriptions via code
    print("\nStep 3: Updating model parameters and descriptions...")
    model.update_predict_params(
        max_new_tokens={
            "type": "int",
            "default": 150,
            "description": "Maximum number of tokens to generate for the response (e.g. 150)"
        },
        temperature={
            "type": "float",
            "default": 0.7,
            "description": "Sampling temperature. Lower is more deterministic, higher is more creative."
        }
    )
    
    print("\nVerifying updated signature & descriptions:")
    model.help()

    # Step 4: Export to a clean JSON file for sharing
    export_path = "facebook-vlm-final-metadata.json"
    print(f"\nStep 4: Exporting model metadata to: {export_path}")
    model.export(export_path)

    # Clean up exported file after test
    if os.path.exists(export_path):
        os.remove(export_path)
        print(f"Cleanup: Removed {export_path}")

if __name__ == "__main__":
    main()
