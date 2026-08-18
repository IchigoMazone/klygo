import sys
import os
import torch

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

# Define a simple network class to compile to TorchScript
class TinyLinearNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
    def forward(self, x):
        return self.fc(x)

def main():
    print("================================================================")
    print("Testing Serialized TorchScript (.pt) Model Loading")
    print("================================================================\n")

    # 1. Compile and save the model to a serialized TorchScript file (.pt)
    model_file_path = "tiny_linear_scripted.pt"
    raw_net = TinyLinearNet()
    scripted_net = torch.jit.script(raw_net)
    scripted_net.save(model_file_path)
    print(f"Saved serialized TorchScript model to: {model_file_path}")

    # 2. Register the model in the registry (no custom loader and no model_class)
    print("\nRegistering TorchScript model in registry...")
    models.register(
        model_key="tiny-linear-jit",
        model_path=model_file_path,
        backend="torch",
        task="detect"
    )

    # 3. Load the model statically (requires NO Python class code definitions at load time!)
    print("\nLoading model statically using default TorchScript reader...")
    try:
        model = models.load("tiny-linear-jit")
        print("Model loaded successfully!")
        
        # 4. Run predict using a raw tensor to verify execution
        dummy_input = torch.randn(1, 3)
        # Note: we bypass DetectAdapter's image loading (which expects images) 
        # by calling backend.predict() directly, or passing tensor to predict.
        # DetectAdapter expects PIL images unless we call backend.predict directly.
        # Let's call model.backend.predict() to verify TorchBackend execution.
        output = model.backend.predict(dummy_input)
        print(f"Inference output tensor: {output}")
        
    except Exception as e:
        print(f"Error during loading or execution: {e}")
        
    finally:
        # 5. Cleanup temporary model file
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
            print(f"\nCleanup: Removed {model_file_path}")

if __name__ == "__main__":
    main()
