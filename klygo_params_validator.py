import json
from typing import Any, Dict, List, Optional
try:
    from pydantic import BaseModel as PydanticBaseModel, Field, ValidationError
except ImportError:
    raise ImportError("This parameter validation demo requires the 'pydantic' package. Please install it via 'pip install pydantic'.")

# =====================================================================
# 1. PARAMETER SCHEMAS DEFINITION (Type-Safe & Self-Documenting)
# =====================================================================

class GroundingPredictParams(PydanticBaseModel):
    """
    Standard parameters schema for Zero-Shot Grounding / Object Detection models
    such as Grounding DINO or NVIDIA LocateAnything.
    """
    text_prompt: str = Field(
        ..., 
        description="Natural language query containing target classes separated by dots (e.g. 'cat . dog')"
    )
    box_threshold: float = Field(
        0.3, 
        ge=0.0, 
        le=1.0, 
        description="Minimum score threshold to filter bounding box predictions (0.0 to 1.0)"
    )
    text_threshold: float = Field(
        0.25, 
        ge=0.0, 
        le=1.0, 
        description="Minimum text similarity alignment threshold (0.0 to 1.0)"
    )


class YoloPredictParams(PydanticBaseModel):
    """
    Standard parameters schema for standard YOLO object detectors.
    """
    conf: float = Field(
        0.25, 
        ge=0.0, 
        le=1.0, 
        description="Object confidence threshold for detection (0.0 to 1.0)"
    )
    iou: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS)"
    )
    verbose: bool = Field(
        False, 
        description="Whether to print inference details to console output"
    )

# =====================================================================
# 2. BASE MODEL WITH DYNAMIC VALIDATION & DOCUMENTATION
# =====================================================================

class BaseModel:
    # Subclasses bind their parameter schema classes here
    predict_params_schema: Optional[PydanticBaseModel] = None
    train_params_schema: Optional[PydanticBaseModel] = None

    def __init__(self, model_key: str, backend: str, task: str):
        self.model_key = model_key
        self.backend = backend
        self.task = task

    def predict(self, source: Any, **kwargs) -> Any:
        """
        Public prediction entry point. Automatically validates input keywords against
        the model's registered predict parameter schema before running inference.
        """
        run_params = kwargs
        
        # If model has a schema defined, perform type checking and validation
        if self.predict_params_schema is not None:
            try:
                # Pydantic automatically checks types, required parameters, and ranges
                validated = self.predict_params_schema(**kwargs)
                # Convert back to clean python dict with all defaults populated
                run_params = validated.model_dump()
            except ValidationError as e:
                # Format validation errors for readable developer output
                error_msg = self._format_validation_error(e)
                raise ValueError(error_msg)

        # Run actual internal prediction
        return self._predict(source, **run_params)

    def _predict(self, source: Any, **kwargs) -> Any:
        # Implemented by sub-adapters
        raise NotImplementedError

    def help(self):
        """
        Prints a beautiful, self-documenting console printout of the parameter schema.
        """
        print(f"============================================================")
        print(f"MODEL: {self.model_key} (backend: {self.backend}, task: {self.task})")
        print(f"============================================================")
        
        if self.predict_params_schema is None:
            print("\n[Predict Parameters (predict())]:")
            print("  No parameters schema registered.")
        else:
            print("\n[Predict Parameters (predict())]:")
            # Get Pydantic properties
            properties = self.predict_params_schema.model_json_schema().get("properties", {})
            required_fields = self.predict_params_schema.model_json_schema().get("required", [])
            
            for param, details in properties.items():
                req_flag = " (Required)" if param in required_fields else ""
                type_name = details.get("type", "Any")
                default_val = details.get("default", None)
                default_str = f", Default: {default_val}" if default_val is not None else ""
                desc = details.get("description", "No description.")
                
                print(f"  * {param} ({type_name}){req_flag}{default_str}")
                print(f"    Description: {desc}")
        print(f"============================================================\n")

    def get_predict_schema_json(self) -> str:
        """Returns standard JSON schema of parameters for API integrations."""
        if self.predict_params_schema:
            return json.dumps(self.predict_params_schema.model_json_schema(), indent=2)
        return "{}"

    def _format_validation_error(self, e: ValidationError) -> str:
        errors = e.errors()
        lines = [f"Validation failed for model '{self.model_key}' parameters:"]
        for err in errors:
            loc = " -> ".join(map(str, err["loc"]))
            msg = err["msg"]
            lines.append(f"  - Parameter '{loc}': {msg}")
        return "\n".join(lines)

# =====================================================================
# 3. TASK ADAPTERS BINDING SCHEMA CLASSES
# =====================================================================

class ZeroShotDetector(BaseModel):
    predict_params_schema = GroundingPredictParams

    def _predict(self, source: Any, **kwargs) -> Any:
        print(f"[{self.model_key}] Inference running on '{source}' with parameters: {kwargs}")
        return {"status": "success", "outputs": []}


class YoloDetector(BaseModel):
    predict_params_schema = YoloPredictParams

    def _predict(self, source: Any, **kwargs) -> Any:
        print(f"[{self.model_key}] YOLO inference running on '{source}' with parameters: {kwargs}")
        return {"status": "success", "outputs": []}

# =====================================================================
# 4. RUN DEMO & VERIFICATION SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("--- STARTING PARAMETERS SCHEMA & VALIDATION DEMO ---\n")
    
    # 1. Instantiate the Zero-Shot Detector (e.g. LocateAnything)
    model = ZeroShotDetector(
        model_key="nvidia-locate-anything", 
        backend="huggingface", 
        task="detect"
    )

    # 2. View model parameters guide in the console
    model.help()

    # 3. TEST A: Call predict with CORRECT arguments
    print("TEST A: Calling predict() with valid arguments:")
    try:
        # We pass only text_prompt. box_threshold and text_threshold are omitted 
        # and should automatically resolve to their default values (0.3 and 0.25).
        model.predict("test_image.jpg", text_prompt="red box . green ball")
        print("  --> Test A: SUCCESS!\n")
    except Exception as err:
        print(f"  --> Test A Failed: {err}\n")

    # 4. TEST B: Call predict with MISSING REQUIRED arguments
    print("TEST B: Calling predict() with missing required 'text_prompt':")
    try:
        model.predict("test_image.jpg")
    except Exception as err:
        print(err)
        print("  --> Test B: SUCCESS (Caught expected error)!\n")

    # 5. TEST C: Call predict with INVALID argument ranges
    print("TEST C: Calling predict() with invalid box_threshold range (e.g. 1.5):")
    try:
        model.predict("test_image.jpg", text_prompt="car", box_threshold=1.5)
    except Exception as err:
        print(err)
        print("  --> Test C: SUCCESS (Caught expected error)!\n")

    # 6. TEST D: Export JSON Schema
    print("TEST D: Exporting standard JSON Schema representation:")
    print(model.get_predict_schema_json())
    print("  --> Test D: SUCCESS!\n")
