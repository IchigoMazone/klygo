import sys
import os
from PIL import Image, ImageDraw

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models
from klygo import media

def main():
    print("Initializing Grounding DINO ('grounding-dino-tiny') zero-shot detector...")
    
    # 1. Load the model from registry (or download online if running for the first time)
    # The task 'detect' is resolved automatically from registration info.
    # Note: Hugging Face transformers model will download raw weights from:
    # https://huggingface.co/IDEA-Research/grounding-dino-tiny
    model = models.load("grounding-dino-tiny")
    
    print("\nModel loaded successfully. Performing warmup...")
    # Optional warmup to compile/load parameters onto GPU
    model.warmup()
    
    # 2. Create a dummy image with a drawn red rectangle and green circle to locate zero-shot
    print("\nGenerating a synthetic test image with shapes...")
    img = Image.new("RGB", (800, 800), color="white")
    draw = ImageDraw.Draw(img)
    # Draw a red rectangle (representing a box/cube)
    draw.rectangle([150, 200, 350, 400], fill="red")
    # Draw a green circle (representing a ball/sphere)
    draw.ellipse([500, 450, 650, 600], fill="green")
    
    temp_img_path = "temp_synthetic_shapes.jpg"
    img.save(temp_img_path)
    print(f"Saved test image to: {temp_img_path}")
    
    # 3. Predict coordinates of target objects using text prompt query (locate anything)
    text_query = "red box . green ball"
    print(f"\nQuerying zero-shot objects with prompt: '{text_query}'")
    
    try:
        results = model.predict(temp_img_path, text_prompt=text_query, box_threshold=0.3, text_threshold=0.25)
        
        print("\n=== DETECTION RESULTS ===")
        for idx, obj in enumerate(results.objects):
            print(f"[{idx}] Found Label: '{obj.label}' with score: {obj.score:.4f}")
            print(f"    Bounding Box: ({int(obj.xmin)}, {int(obj.ymin)}) -> ({int(obj.xmax)}, {int(obj.ymax)})")
            
    except Exception as e:
        print(f"\nPrediction failed: {e}")
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            print(f"Removed temp image {temp_img_path}")
            
    # 4. Unload model to free GPU VRAM
    print("\nUnloading model weights from GPU memory...")
    model.unload()

if __name__ == "__main__":
    main()
