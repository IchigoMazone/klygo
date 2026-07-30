import base64
import zlib

BACKEND_MAP = {"huggingface": "h", "ultralytics": "u", "torch": "t"}
TASK_MAP = {
    "detect": "d", 
    "classify": "c", 
    "segment": "s", 
    "llm": "l", 
    "embedding": "e", 
    "speech": "p", 
    "ocr": "o"
}

REV_BACKEND_MAP = {v: k for k, v in BACKEND_MAP.items()}
REV_TASK_MAP = {v: k for k, v in TASK_MAP.items()}

def encode_registry_key(model_key: str, model_path: str, backend: str, task: str) -> str:
    """
    Compresses and encodes model metadata including the model_key into a 64-character Base64 token.
    """
    b_code = BACKEND_MAP.get(backend, "u")
    t_code = TASK_MAP.get(task, "d")
    
    # Raw representation format: model_key|backend_code|task_code|model_path
    raw_str = f"{model_key}|{b_code}|{t_code}|{model_path}"
    
    compressed = zlib.compress(raw_str.encode('utf-8'))
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8').rstrip('=')
    
    # Ensure it is padded to exactly 64 characters
    if len(encoded) < 64:
        encoded = encoded.ljust(64, '_')
    elif len(encoded) > 64:
        raise ValueError("Model path or key is too long to fit in a 64-character encoded token.")
        
    return encoded

def decode_registry_key(key: str) -> dict:
    """
    Decodes the 64-character token back into structured metadata.
    """
    clean_key = key.rstrip('_')
    padding = len(clean_key) % 4
    if padding:
        clean_key += '=' * (4 - padding)
        
    compressed = base64.urlsafe_b64decode(clean_key.encode('utf-8'))
    raw_str = zlib.decompress(compressed).decode('utf-8')
    
    model_key, b_code, t_code, model_path = raw_str.split('|', 3)
    
    return {
        "model_key": model_key,
        "backend": REV_BACKEND_MAP.get(b_code, "ultralytics"),
        "task": REV_TASK_MAP.get(t_code, "detect"),
        "model_path": model_path
    }
