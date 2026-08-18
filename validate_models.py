import os
import json
import inspect
from typing import Any, Dict, Optional

# Path to the decentralized registry directory
REGISTRY_DIR = os.path.expanduser("~/.klygo/registry")
os.makedirs(REGISTRY_DIR, exist_ok=True)

# =====================================================================
# 1. METADATA HELPERS: DYNAMIC SIGNATURE & DOCSTRING BINDING (IDE DX)
# =====================================================================

def bind_dynamic_ide_metadata(predict_method: Any, schema: Dict[str, Any], guide: str, links: dict):
    """
    Dynamically binds parameter signatures and docstrings to the predict method.
    This enables autocomplete and description popups directly in modern IDEs.
    """
    # 1. Generate Python signature
    new_params = [
        inspect.Parameter("source", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any)
    ]
    type_mapping = {"str": str, "float": float, "int": int, "bool": bool}
    
    for param_name, info in schema.items():
        expected_type = type_mapping.get(info.get("type"), Any)
        if info.get("required"):
            default_val = inspect.Parameter.empty
        else:
            default_val = info.get("default", None)
            
        new_params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default_val,
                annotation=expected_type
            )
        )
    
    # Apply signature to the predict function
    predict_method.__func__.__signature__ = inspect.Signature(new_params)
    
    # 2. Generate Docstring
    doc_lines = [
        f"[Guide]: {guide}",
        ""
    ]
    if links:
        doc_lines.append("[Documentation & Tutorials]:")
        for title, url in links.items():
            doc_lines.append(f"  * {title.replace('_', ' ').capitalize()}: {url}")
        doc_lines.append("")
        
    doc_lines.append("Parameters:")
    doc_lines.append("-----------")
    for param_name, info in schema.items():
        req = " (Required)" if info.get("required") else ""
        default = f", default: {info.get('default')}" if "default" in info else ""
        doc_lines.append(f"{param_name} : {info.get('type')}{req}{default}")
        doc_lines.append(f"    {info.get('description', 'No description.')}")
        
    # Apply docstring to the predict function
    predict_method.__func__.__doc__ = "\n".join(doc_lines)

# =====================================================================
# 2. BASE MODEL WITH DECENTRALIZED METADATA & PARAMETER VALIDATION
# =====================================================================

class BaseModel:
    def __init__(self, model_key: str, metadata: dict):
        self.model_key = model_key
        self.metadata = metadata
        self.predict_params = metadata.get("predict_params", {})
        
        # Bind signature and help docstring dynamically
        bind_dynamic_ide_metadata(
            predict_method=self.predict,
            schema=self.predict_params,
            guide=metadata.get("guide", "No guide available."),
            links=metadata.get("links", {})
        )

    def predict(self, source: Any, **kwargs) -> Any:
        """
        Public inference method. Automatically validates input arguments,
        type-casts parameters, and injects default values from the schema.
        """
        # Validate parameters and inject defaults based on schema
        for param_name, info in self.predict_params.items():
            if info.get("required") and param_name not in kwargs:
                raise ValueError(f"Parameter '{param_name}' is required for model '{self.model_key}' but was not provided.")
            
            if param_name not in kwargs and "default" in info:
                # Inject default value
                kwargs[param_name] = info["default"]
            elif param_name in kwargs:
                # Type check and safe cast
                expected_type_str = info.get("type", "Any")
                val = kwargs[param_name]
                type_mapping = {"str": str, "float": float, "int": int, "bool": bool}
                expected_type = type_mapping.get(expected_type_str)
                if expected_type and not isinstance(val, expected_type):
                    try:
                        kwargs[param_name] = expected_type(val)
                    except Exception:
                        raise TypeError(f"Parameter '{param_name}' must be of type {expected_type_str}, got {type(val).__name__} instead.")

        return self._predict(source, **kwargs)

    def _predict(self, source: Any, **kwargs) -> Any:
        # Overridden by task adapters
        raise NotImplementedError

    def update_predict_params(self, **updated_schema):
        """
        Saves/updates advanced parameter definitions directly to the model's 
        individual decentralized JSON metadata file.
        """
        filepath = os.path.join(REGISTRY_DIR, f"{self.model_key}.json")
        
        # Merge changes
        self.metadata["predict_params"].update(updated_schema)
        self.predict_params = self.metadata["predict_params"]
        
        # Write back to individual JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
            
        print(f"-> Successfully saved updated metadata to: {filepath}")
        
        # Re-bind metadata to reflect updates in current session
        bind_dynamic_ide_metadata(
            predict_method=self.predict,
            schema=self.predict_params,
            guide=self.metadata.get("guide", "No guide available."),
            links=self.metadata.get("links", {})
        )

    def export_metadata(self, target_path: str):
        """Exports the configuration file in standard JSON format for community sharing."""
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        print(f"-> Exported model metadata for sharing to: {target_path}")

    def help(self):
        """Prints a beautifully formatted user guide to console."""
        print(f"\n====================== HELP: {self.model_key} ======================")
        print(self.predict.__doc__)
        print("====================================================================")

# =====================================================================
# 3. DYNAMIC PARAMETERS DISCOVERY & MODEL LOAD ENGINE
# =====================================================================

def discover_params_from_backend(raw_model: Any) -> dict:
    """Uses reflection to dynamically extract arguments and defaults from backend function."""
    import inspect
    discovered = {}
    INTERNAL_PARAMS = {"self", "images", "text", "audio", "return_tensors", "args", "kwargs"}
    
    sigs = []
    # Quét signature của generate() hoặc forward()
    if hasattr(raw_model, "generate"):
        try:
            sigs.append(inspect.signature(raw_model.generate))
        except Exception:
            pass
    elif hasattr(raw_model, "forward"):
        try:
            sigs.append(inspect.signature(raw_model.forward))
        except Exception:
            pass
            
    for sig in sigs:
        for name, param in sig.parameters.items():
            if name not in INTERNAL_PARAMS:
                type_name = "Any"
                if param.annotation is not inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        type_name = param.annotation.__name__
                    else:
                        type_name = str(param.annotation)
                discovered[name] = {
                    "type": type_name,
                    "required": param.default is inspect.Parameter.empty,
                    "default": param.default if param.default is not inspect.Parameter.empty else None,
                    "description": "Dynamically discovered parameter."
                }
    return discovered

def load_model(model_key: str, raw_model_obj: Any = None) -> BaseModel:
    """Loads model from separate JSON file, falling back to dynamic signature discovery."""
    filepath = os.path.join(REGISTRY_DIR, f"{model_key}.json")
    
    # 1. Load from decentralized JSON file
    if os.path.exists(filepath):
        print(f"-> Loading decentralized config from: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    # 2. Fallback: Dynamic Parameter Discovery
    else:
        print(f"-> Model '{model_key}' has no config file. Running Dynamic Signature Discovery...")
        discovered_params = discover_params_from_backend(raw_model_obj)
        metadata = {
            "model_key": model_key,
            "backend": "torch",
            "task": "detect",
            "guide": "Custom model with auto-discovered parameter signatures.",
            "predict_params": discovered_params
        }
        # Write initial metadata to individual file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
    # Mock runtime Adapter
    class MockAdapter(BaseModel):
        def _predict(self, source, **kwargs):
            return f"Mock inference on '{source}' using variables: {kwargs}"
            
    return MockAdapter(model_key, metadata)

# =====================================================================
# 4. RUN DEMO OF ALL CASES
# =====================================================================

if __name__ == "__main__":
    # Clean previous runs
    import shutil
    shutil.rmtree(REGISTRY_DIR, ignore_errors=True)
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    
    print("--- STARTING DECENTRALIZED VALIDATOR TEST SUITE ---\n")
    
    # -----------------------------------------------------------------
    # Case 1: Load a brand-new model (Triggers Dynamic Discovery)
    # -----------------------------------------------------------------
    print("1. LOADING NEW MODEL FOR THE FIRST TIME:")
    
    # We define a custom network class with custom signatures
    import torch
    class CustomTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def generate(self, input_ids, max_new_tokens: int = 150, temperature: float = 0.8):
            pass
            
    # Load model
    model = load_model("my-custom-vlm", raw_model_obj=CustomTorchModel())
    
    # Verify signature popup help:
    model.help()
    
    # -----------------------------------------------------------------
    # Case 2: Run predictions (Verifies defaults & type checking)
    # -----------------------------------------------------------------
    print("\n2. VERIFYING RUNTIME PREDICTIONS:")
    
    # Try calling with valid parameters
    result = model.predict("image.jpg", input_ids=[1, 2, 3], temperature=0.7)
    print(f"Success call: {result}")
    
    # Try calling with invalid types (e.g. max_new_tokens = 'high')
    try:
        model.predict("image.jpg", input_ids=[1, 2, 3], max_new_tokens="high")
    except TypeError as e:
        print(f"Caught expected type error: {e}")
        
    # -----------------------------------------------------------------
    # Case 3: Developer/User updates advanced description & locks params
    # -----------------------------------------------------------------
    print("\n3. UPDATING PARAMETERS SCHEMA WITH DETAILED DESCRIPTIONS:")
    
    model.update_predict_params(
        max_new_tokens={
            "type": "int",
            "default": 150,
            "description": "Maximum number of response tokens to generate (e.g. 150)"
        },
        temperature={
            "type": "float",
            "default": 0.8,
            "description": "Sampling temperature. Lower is more deterministic, higher is more creative."
        }
    )
    
    # Verify the help is updated with the detailed descriptions
    model.help()
    
    # -----------------------------------------------------------------
    # Case 4: Subsequent Load (Loads the updated custom config file)
    # -----------------------------------------------------------------
    print("\n4. LOADING MODEL AGAIN IN NEXT PROJECT RUNS:")
    reloaded_model = load_model("my-custom-vlm")
    reloaded_model.help()
    
    # Export for sharing
    reloaded_model.export_metadata("shared_model_metadata.json")
