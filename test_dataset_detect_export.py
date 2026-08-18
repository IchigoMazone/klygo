import sys
import os
import shutil
from PIL import Image

# Add the parent directory of klygo to python path to import klygo successfully
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from klygo import models
from klygo import datasets

def main():
    print("================================================================")
    print("Testing New Module API: datasets.detect.export()")
    print("================================================================\n")

    # 1. Setup mock images directory
    source_dir = "raw_test_images"
    os.makedirs(source_dir, exist_ok=True)
    dummy_img_path = os.path.join(source_dir, "test_image.jpg")
    
    # Save a 100x100 dummy black image
    img = Image.new("RGB", (100, 100), color="black")
    img.save(dummy_img_path)
    print(f"Created temporary image for testing: {dummy_img_path}")

    # 2. Register mock detection model
    from klygo.models.registry import registry
    import torch
    
    class MockDetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return None

    @registry.register_loader_fn("my_mock_loader")
    def my_mock_loader(model_path):
        return MockDetectionModel()

    models.register(
        model_key="mock-vlm-detect",
        model_path="facebook/brand-new-vlm",
        backend="torch",
        task="detect",
        loader="my_mock_loader"
    )
    
    # Load model
    model = models.load("mock-vlm-detect")
    
    # 3. Call the datasets.detect.export() function directly!
    dataset_output = "exported_yolo_dataset"
    print("\nCalling datasets.detect.export()...")
    try:
        datasets.detect.export(
            model=model,
            output_path=dataset_output,
            format="yolo",
            source=source_dir,
            classes=["cat", "dog"],
            verbose=False # Keep output clean
        )
        print("Export completed successfully!")
        
        # 4. Verify created dataset structures
        yaml_path = os.path.join(dataset_output, "dataset.yaml")
        img_exists = os.path.exists(os.path.join(dataset_output, "images", "img_0.jpg"))
        lbl_exists = os.path.exists(os.path.join(dataset_output, "labels", "img_0.txt"))
        
        print("\nVerifying output structure:")
        print(f"  * dataset.yaml exists: {os.path.exists(yaml_path)}")
        print(f"  * images/img_0.jpg exists: {img_exists}")
        print(f"  * labels/img_0.txt exists: {lbl_exists}")
        
        if os.path.exists(yaml_path) and img_exists and lbl_exists:
            print("\nRESULT: Success! datasets.detect.export() works perfectly.")
        else:
            print("\nRESULT: Failed to verify output structure.")
            
    except Exception as e:
        print(f"Error during dataset export execution: {e}")
        
    finally:
        # 5. Cleanup temporary folders
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(dataset_output, ignore_errors=True)
        print("\nCleanup: Removed temporary test folders.")

if __name__ == "__main__":
    main()
