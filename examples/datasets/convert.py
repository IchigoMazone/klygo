"""Convert Pascal VOC XML annotations to a flat YOLO detection dataset."""

from tempfile import TemporaryDirectory
from PIL import Image

from klygo import datasets, files, media


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "voc_raw"
    images_dir = source / "JPEGImages"
    ann_dir = source / "Annotations"
    files.mkdir(images_dir)
    files.mkdir(ann_dir)

    # Create a sample image and Pascal VOC XML annotation
    media.save(images_dir / "000001.jpg", Image.new("RGB", (200, 100), "white"), verbose=False)
    xml_content = """<annotation>
    <size>
        <width>200</width>
        <height>100</height>
    </size>
    <object>
        <name>car</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>20</xmin>
            <ymin>10</ymin>
            <xmax>100</xmax>
            <ymax>60</ymax>
        </bndbox>
    </object>
</annotation>"""
    files.save(ann_dir / "000001.xml", xml_content, verbose=False)

    # Convert to flat YOLO dataset
    target = root / "yolo_flat"
    datasets.convert(
        source=source,
        target=target,
        src_format="voc",
        classes=["car"],
    )

    print(datasets.info(target))
