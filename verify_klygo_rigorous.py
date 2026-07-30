import sys
import os
import torch

# Ensure klygo directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import klygo
from klygo import models
from klygo.models.registry import registry
from klygo.models.base import BaseModel
from klygo.models.adapters.classify import ClassificationResult

print("====================================================================")
print("Klygo Models RIGOROUS Production Readiness Tests")
print("====================================================================")

# --------------------------------------------------------------------
# TEST 1: Custom Functional Model Registration (@registry.register_model_fn)
# --------------------------------------------------------------------
print("\n[TEST 1/5] Testing Custom Functional Model Hook...")
@registry.register_model_fn("my-custom-fn-model")
def custom_model_fn(x, **kwargs):
    return f"Functional model output: {x * 10}"

try:
    # Load functional model
    model = models.load("my-custom-fn-model")
    print(f"  - Loaded custom functional model: {model}")
    
    # Run prediction
    res = model.predict(5)
    print(f"  - Predict output: {res}")
    assert res == "Functional model output: 50"
    print("--> TEST 1 PASSED!")
except Exception as e:
    print(f"[FAIL] TEST 1 Failed: {e}")
    sys.exit(1)


# --------------------------------------------------------------------
# TEST 2: Custom Loader & Preprocess/Postprocess Hooks
# --------------------------------------------------------------------
print("\n[TEST 2/5] Testing Custom Loader and Pre/Postprocess Hooks...")

# Define dummy PyTorch model
class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        # Fix weights to identity for deterministic testing
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))
            self.fc.bias.fill_(0.0)
    def forward(self, x):
        return self.fc(x)

# 1. Register custom loader function
@registry.register_loader_fn("test_loader_fn")
def test_loader(model_path):
    print(f"  - Custom loader called with path: {model_path}")
    return DummyNet()

# 2. Register preprocessing hook
@registry.register_preprocess("my-test-hooked-model")
def preprocess_hook(tensor_input, **kwargs):
    print(f"  - Preprocess hook called with input: {tensor_input}")
    if isinstance(tensor_input, list):
        return [t + 10.0 for t in tensor_input]
    return tensor_input + 10.0

# 3. Register postprocessing hook
@registry.register_postprocess("my-test-hooked-model")
def postprocess_hook(classify_res, **kwargs):
    print(f"  - Postprocess hook called with result: {classify_res}")
    if isinstance(classify_res, list):
        for res in classify_res:
            if not res.label.startswith("processed_"):
                res.label = f"processed_{res.label}"
        return classify_res
    if not classify_res.label.startswith("processed_"):
        classify_res.label = f"processed_{classify_res.label}"
    return classify_res

# 4. Register the model using the custom loader
models.register(
    model_key="my-test-hooked-model",
    model_path="dummy_weights.pth",
    backend="torch",
    task="classify",
    loader="test_loader_fn"
)

try:
    # Load hooked model
    model = models.load("my-test-hooked-model", classes=["yes", "no"])
    print(f"  - Loaded hooked model: {model}")
    
    # Run prediction (expecting raw output to be processed by class adapter)
    # Input is [1.0, -1.0]. 
    # Preprocess: [1.0, -1.0] + 10.0 = [11.0, 9.0]
    # Model: [11.0, 9.0] -> outputs logits [11.0, 9.0]
    # ClassifyAdapter: argmax is index 0 -> label "yes"
    # Postprocess: "yes" -> "processed_yes"
    test_tensor = torch.tensor([1.0, -1.0])
    res = model.predict(test_tensor)
    print(f"  - Final prediction result: {res}")
    
    assert res.label == "processed_yes"
    print("--> TEST 2 PASSED!")
except Exception as e:
    print(f"[FAIL] TEST 2 Failed: {e}")
    sys.exit(1)


# --------------------------------------------------------------------
# TEST 3: Batch Inference (List-in, List-out) & Single Inference (Single-in, Single-out)
# --------------------------------------------------------------------
print("\n[TEST 3/5] Testing Batch vs Single Inference on ClassifyAdapter...")
try:
    # We will use the previously loaded 'my-test-hooked-model' (ClassifyAdapter)
    
    # CASE A: Single Inference (Single Tensor input)
    single_input = torch.tensor([2.0, -2.0]) # Preprocessed: [12.0, 8.0] -> "yes"
    single_res = model.predict(single_input)
    print(f"  - Single predict type: {type(single_res)}")
    assert isinstance(single_res, ClassificationResult)
    assert single_res.label == "processed_yes"
    
    # CASE B: Batch Inference (List of Tensors input)
    batch_input = [torch.tensor([2.0, -2.0]), torch.tensor([-2.0, 2.0])]
    # Preprocessed 1: [12.0, 8.0] -> "yes"
    # Preprocessed 2: [8.0, 12.0] -> "no"
    batch_res = model.predict(batch_input)
    print(f"  - Batch predict type: {type(batch_res)}")
    assert isinstance(batch_res, list)
    assert len(batch_res) == 2
    assert batch_res[0].label == "processed_yes"
    assert batch_res[1].label == "processed_no"
    
    print("--> TEST 3 PASSED!")
except Exception as e:
    print(f"[FAIL] TEST 3 Failed: {e}")
    sys.exit(1)


# --------------------------------------------------------------------
# TEST 4: GPU serving APIs (Warmup and Unload)
# --------------------------------------------------------------------
print("\n[TEST 4/5] Testing GPU serving Warmup & Unload methods...")
try:
    # Test warmup
    model.warmup()
    
    # Test unload
    model.unload()
    print("--> TEST 4 PASSED!")
except Exception as e:
    print(f"[FAIL] TEST 4 Failed: {e}")
    sys.exit(1)


# --------------------------------------------------------------------
# TEST 5: Path Overwrite Fallback
# --------------------------------------------------------------------
print("\n[TEST 5/5] Testing Path Override Fallback...")
try:
    # Force load using a local directory mock to test offline path redirection
    dummy_offline_path = "offline_dummy.pth"
    
    # Write a temporary state dict file to load
    torch.save(DummyNet().state_dict(), dummy_offline_path)
    
    # Register online path key
    models.register(
        model_key="online-model-mock",
        model_path="http://cdn.klygo.com/online_weights.pth",
        backend="torch",
        task="classify"
    )
    
    # Load and override path to local file (bypasses online CDN url)
    loaded_model = models.load(
        "online-model-mock",
        model_path=dummy_offline_path,
        model_class=DummyNet,
        classes=["yes", "no"]
    )
    
    print(f"  - Model path resolved: {loaded_model.config['model_path']}")
    assert loaded_model.config["model_path"] == dummy_offline_path
    
    # Cleanup file
    if os.path.exists(dummy_offline_path):
        os.remove(dummy_offline_path)
        
    print("--> TEST 5 PASSED!")
except Exception as e:
    print(f"[FAIL] TEST 5 Failed: {e}")
    if os.path.exists(dummy_offline_path):
        os.remove(dummy_offline_path)
    sys.exit(1)

print("\n====================================================================")
print("All Rigorous Production Tests Completed Successfully!")
print("====================================================================")
