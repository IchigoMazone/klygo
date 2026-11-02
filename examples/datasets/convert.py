"""Convert multiple raw source trees into one flat YOLO dataset."""

from PIL import Image

from klygo import datasets, files, media


root = files.join("artifacts", "examples", "datasets", "convert")
source = files.join(root, "raw")
target = files.join(root, "flat")
files.remove(root, missing_ok=True)
for batch, stem in (("batch_a", "apple_1"), ("batch_b", "banana_1")):
    images = files.join(source, batch, "images")
    labels = files.join(source, batch, "labels")
    files.mkdir(images)
    files.mkdir(labels)
    media.save(files.join(images, stem + ".jpg"), Image.new("RGB", (64, 48), "white"), verbose=False)
    files.save(files.join(labels, stem + ".txt"), ["0 0.5 0.5 0.4 0.4"], verbose=False)

output = datasets.convert(
    source,
    target,
    tree={"pictures": "batch_*/images", "annotations": "batch_*/labels"},
    classes={0: "fruit"},
    on_conflict="rename",
)
print(datasets.info(output, verbose=False))
