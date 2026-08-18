import sys
import os
from PIL import Image, ImageDraw

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models

def main():
    print("================================================================")
    # NVIDIA LocateAnything-3B requires Python libraries like decord, lmdb, peft, and torch
    print("Initializing NVIDIA LocateAnything-3B visual grounding VLM...")
    print("================================================================")

    # 1. Register NVIDIA LocateAnything-3B model
    # We specify task="detect" and backend="huggingface"
    model_key = "nvidia-locate-anything"
    models.register(
        model_key=model_key,
        model_path="nvidia/LocateAnything-3B",
        backend="huggingface",
        task="detect"
    )
    print(f"Model '{model_key}' registered successfully.")

    # 2. Load the model
    # Note: LocateAnything requires trust_remote_code=True
    # Our package backend automatically defaults trust_remote_code=True for locateanything, 
    # but we can also pass it explicitly.
    print(f"Loading '{model_key}' (this will fetch model from Hugging Face if not cached)...")
    try:
        model = models.load(model_key, trust_remote_code=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nNote: Make sure you have transformers, torch, peft, lmdb, and decord installed.")
        return

    # 3. Create a test image
    print("\nGenerating a test image with distinct shapes...")
    img = Image.new("RGB", (600, 600), color="white")
    draw = ImageDraw.Draw(img)
    # Draw a blue rectangle (representing a blue box)
    draw.rectangle([100, 100, 250, 250], fill="blue")
    # Draw a red circle (representing a red ball)
    draw.ellipse([350, 350, 480, 480], fill="red")
    
    test_image_path = "temp_nvidia_test.jpg"
    img.save(test_image_path)
    print(f"Test image saved as '{test_image_path}'.")

    # 4. Predict (Locate red ball and blue box)
    prompt = "Locate the red ball and the blue box."
    print(f"\nQuerying: '{prompt}'")
    try:
        results = model.predict(test_image_path, text_prompt=prompt)
        
        print("\n=== DETECTED BOXES ===")
        for idx, obj in enumerate(results.objects):
            print(f"[{idx}] Object: '{obj.label}'")
            print(f"    Confidence score: {obj.score:.2f}")
            print(f"    Bounding Box (pixel coords): ({int(obj.xmin)}, {int(obj.ymin)}) -> ({int(obj.xmax)}, {int(obj.ymax)})")
            
    except Exception as e:
        print(f"Prediction failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Removed '{test_image_path}'")

    # 5. Unload VRAM
    print("\nUnloading weights from GPU...")
    model.unload()

if __name__ == "__main__":
    main()
