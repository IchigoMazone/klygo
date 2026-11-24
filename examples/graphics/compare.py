"""Compare two Detection frames and save without opening a viewer."""
from PIL import Image
from klygo import files, outputs, graphics

first = outputs.Detection(
    [outputs.Box((40, 50, 140, 170), label="cat", score=0.9)],
    source=Image.new("RGB", (320, 240), "lightgray"),
    source_id="image-a", image_size=(320, 240),
)
second = outputs.Detection(
    [outputs.Box((70, 40, 190, 150), label="dog", score=0.8)],
    source=Image.new("RGB", (280, 200), "lightblue"),
    source_id="image-b", image_size=(280, 200),
)
output = files.path("artifacts/examples/graphics/comparison.png")
preview = graphics.compare(
    first, second, mode="save", output_path=output,
    titles=("Image A", "Image B"), overwrite=True,
)
print(output, preview.size)
