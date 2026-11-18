"""Tests for annotation conversion datasets.convert()."""

import tempfile
import unittest
from PIL import Image

from klygo import config, datasets, files, media


class ConvertAnnotationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = files.path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_sample_image(self, path, width: int = 200, height: int = 100, color: str = "red"):
        files.mkdir(files.parent(path))
        img = Image.new("RGB", (width, height), color)
        media.save(path, img, overwrite=True, verbose=False)
        return files.hash(path, algorithm="sha256")

    def test_voc_xml_conversion_and_round_trip(self) -> None:
        voc_dir = files.join(self.root, "voc_raw")
        img_dir = files.join(voc_dir, "JPEGImages")
        ann_dir = files.join(voc_dir, "Annotations")
        files.mkdir(img_dir)
        files.mkdir(ann_dir)

        # Image 1: 200x100 with two objects (one normal 'dog', one difficult 'cat')
        img1_path = files.join(img_dir, "000001.jpg")
        img1_hash = self._create_sample_image(img1_path, 200, 100)
        xml1_content = """<annotation>
    <size>
        <width>200</width>
        <height>100</height>
        <depth>3</depth>
    </size>
    <object>
        <name>dog</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>20</xmin>
            <ymin>10</ymin>
            <xmax>100</xmax>
            <ymax>60</ymax>
        </bndbox>
    </object>
    <object>
        <name>cat</name>
        <difficult>1</difficult>
        <bndbox>
            <xmin>120</xmin>
            <ymin>30</ymin>
            <xmax>180</xmax>
            <ymax>90</ymax>
        </bndbox>
    </object>
</annotation>"""
        files.save(files.join(ann_dir, "000001.xml"), xml1_content, verbose=False)

        # Image 2: negative image with no objects
        img2_path = files.join(img_dir, "000002.jpg")
        img2_hash = self._create_sample_image(img2_path, 200, 100, color="blue")
        xml2_content = """<annotation>
    <size>
        <width>200</width>
        <height>100</height>
    </size>
</annotation>"""
        files.save(files.join(ann_dir, "000002.xml"), xml2_content, verbose=False)

        target = files.join(self.root, "yolo_voc")
        out_path = datasets.convert(
            source=voc_dir,
            target=target,
            src_format="voc",
            include_difficult=False,
            classes=["cat", "dog"],
        )
        self.assertIsNone(out_path)

        # Verify image byte preservation
        out_img1 = files.join(target, "images", "000001.jpg")
        out_img2 = files.join(target, "images", "000002.jpg")
        self.assertTrue(files.is_file(out_img1))
        self.assertTrue(files.is_file(out_img2))
        self.assertEqual(files.hash(out_img1, algorithm="sha256"), img1_hash)
        self.assertEqual(files.hash(out_img2, algorithm="sha256"), img2_hash)

        # Verify label contents: dog is class 1 ('cat' is 0, 'dog' is 1)
        # 'cat' was difficult so excluded by include_difficult=False
        lbl1 = files.load(files.join(target, "labels", "000001.txt"), as_lines=True, verbose=False)
        self.assertEqual(len(lbl1), 1)
        parts = lbl1[0].split()
        cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        self.assertEqual(cid, 1)

        # Check coordinate values: xmin=20, xmax=100 in 200 width -> xc = 60/200 = 0.3, bw = 80/200 = 0.4
        # ymin=10, ymax=60 in 100 height -> yc = 35/100 = 0.35, bh = 50/100 = 0.5
        self.assertAlmostEqual(xc, 0.30, places=5)
        self.assertAlmostEqual(yc, 0.35, places=5)
        self.assertAlmostEqual(bw, 0.40, places=5)
        self.assertAlmostEqual(bh, 0.50, places=5)

        # Round trip check back to pixels
        xmin_rec = (xc - bw / 2.0) * 200
        ymin_rec = (yc - bh / 2.0) * 100
        xmax_rec = (xc + bw / 2.0) * 200
        ymax_rec = (yc + bh / 2.0) * 100
        self.assertAlmostEqual(xmin_rec, 20.0, places=3)
        self.assertAlmostEqual(ymin_rec, 10.0, places=3)
        self.assertAlmostEqual(xmax_rec, 100.0, places=3)
        self.assertAlmostEqual(ymax_rec, 60.0, places=3)

        # Verify negative image label is empty
        lbl2 = files.load(files.join(target, "labels", "000002.txt"), as_lines=True, verbose=False)
        self.assertEqual(len(lbl2), 0)

        # Verify data.yaml
        meta = config.load(files.join(target, "data.yaml"), verbose=False).to_dict()
        self.assertEqual(meta["nc"], 2)
        self.assertEqual(meta["names"], {0: "cat", 1: "dog"})

    def test_voc_1_based_offset(self) -> None:
        voc_dir = files.join(self.root, "voc_1based")
        img_dir = files.join(voc_dir, "JPEGImages")
        ann_dir = files.join(voc_dir, "Annotations")
        files.mkdir(img_dir)
        files.mkdir(ann_dir)

        img_path = files.join(img_dir, "one.jpg")
        self._create_sample_image(img_path, 100, 100)
        xml_content = """<annotation>
    <size><width>100</width><height>100</height></size>
    <object>
        <name>box</name>
        <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>51</xmax><ymax>51</ymax></bndbox>
    </object>
</annotation>"""
        files.save(files.join(ann_dir, "one.xml"), xml_content, verbose=False)

        target = files.join(self.root, "yolo_1based")
        datasets.convert(source=voc_dir, target=target, src_format="voc", voc_1_based=True)

        lbl = files.load(files.join(target, "labels", "one.txt"), as_lines=True, verbose=False)
        parts = lbl[0].split()
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # xmin was 1 -> 0, xmax was 51 -> 51 => bw = 51/100 = 0.51, xc = (0 + 51)/200 = 0.255
        self.assertAlmostEqual(xc, 0.255, places=5)
        self.assertAlmostEqual(yc, 0.255, places=5)
        self.assertAlmostEqual(bw, 0.51, places=5)
        self.assertAlmostEqual(bh, 0.51, places=5)

    def test_coco_json_conversion_sparse_categories(self) -> None:
        coco_dir = files.join(self.root, "coco_raw")
        img_dir = files.join(coco_dir, "images")
        files.mkdir(img_dir)

        # Two images: one with detections, one negative
        img1_path = files.join(img_dir, "0001.jpg")
        img1_hash = self._create_sample_image(img1_path, 400, 200)

        img2_path = files.join(img_dir, "0002.jpg")
        img2_hash = self._create_sample_image(img2_path, 400, 200, color="green")

        # COCO JSON with sparse category IDs: 5 (bicycle) and 12 (car)
        coco_json = {
            "images": [
                {"id": 101, "file_name": "0001.jpg", "width": 400, "height": 200},
                {"id": 102, "file_name": "0002.jpg", "width": 400, "height": 200},
            ],
            "categories": [
                {"id": 12, "name": "car"},
                {"id": 5, "name": "bicycle"},
            ],
            "annotations": [
                # Normal box: x=40, y=20, w=100, h=60 (category 12 -> car)
                {"id": 1, "image_id": 101, "category_id": 12, "bbox": [40, 20, 100, 60], "iscrowd": 0},
                # Crowd box: should be excluded by default
                {"id": 2, "image_id": 101, "category_id": 5, "bbox": [150, 50, 50, 50], "iscrowd": 1},
            ],
        }
        ann_file = files.join(coco_dir, "instances.json")
        files.save(ann_file, coco_json, verbose=False)

        target = files.join(self.root, "yolo_coco")
        datasets.convert(
            source=coco_dir,
            target=target,
            src_format="coco",
            include_crowd=False,
        )

        # Category sorting by id: 5 (bicycle) -> 0, 12 (car) -> 1
        meta = config.load(files.join(target, "data.yaml"), verbose=False).to_dict()
        self.assertEqual(meta["names"], {0: "bicycle", 1: "car"})

        lbl1 = files.load(files.join(target, "labels", "0001.txt"), as_lines=True, verbose=False)
        self.assertEqual(len(lbl1), 1)  # crowd box excluded
        parts = lbl1[0].split()
        cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        self.assertEqual(cid, 1)  # car is index 1

        # x=40, w=100 -> xc = (40 + 140)/2 / 400 = 0.225, bw = 100/400 = 0.25
        # y=20, h=60 -> yc = (20 + 80)/2 / 200 = 0.25, bh = 60/200 = 0.3
        self.assertAlmostEqual(xc, 0.225, places=5)
        self.assertAlmostEqual(yc, 0.250, places=5)
        self.assertAlmostEqual(bw, 0.250, places=5)
        self.assertAlmostEqual(bh, 0.300, places=5)

        # Image bytes check
        self.assertEqual(files.hash(files.join(target, "images", "0001.jpg"), algorithm="sha256"), img1_hash)
        self.assertEqual(files.hash(files.join(target, "images", "0002.jpg"), algorithm="sha256"), img2_hash)

        # Negative image has empty label
        lbl2 = files.load(files.join(target, "labels", "0002.txt"), as_lines=True, verbose=False)
        self.assertEqual(len(lbl2), 0)

    def test_coco_include_crowd_and_image_ids(self) -> None:
        coco_dir = files.join(self.root, "coco_crowd")
        img_dir = files.join(coco_dir, "images")
        files.mkdir(img_dir)

        self._create_sample_image(files.join(img_dir, "img1.jpg"), 100, 100)
        self._create_sample_image(files.join(img_dir, "img2.jpg"), 100, 100)

        coco_json = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "thing"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 1},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [30, 30, 40, 40], "iscrowd": 0},
            ],
        }
        files.save(files.join(coco_dir, "instances.json"), coco_json, verbose=False)

        target = files.join(self.root, "yolo_crowd")
        datasets.convert(
            source=coco_dir,
            target=target,
            src_format="coco",
            include_crowd=True,
            image_ids=["img1"],  # only convert img1
        )
        info = datasets.info(target)
        self.assertEqual(info["total_images"], 1)
        lbl1 = files.load(files.join(target, "labels", "img1.txt"), as_lines=True, verbose=False)
        self.assertEqual(len(lbl1), 1)

    def test_tree_argument_raises_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "datasets.flatten"):
            datasets.convert(
                source=self.root,
                target=files.join(self.root, "out"),
                tree={"images": "images", "labels": "labels"},
            )

    def test_export_to_zip_archive(self) -> None:
        voc_dir = files.join(self.root, "voc_raw_zip")
        img_dir = files.join(voc_dir, "JPEGImages")
        ann_dir = files.join(voc_dir, "Annotations")
        files.mkdir(img_dir)
        files.mkdir(ann_dir)

        img_path = files.join(img_dir, "sample.jpg")
        self._create_sample_image(img_path, 100, 100)
        xml_content = """<annotation>
    <size><width>100</width><height>100</height></size>
    <object><name>item</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>90</xmax><ymax>90</ymax></bndbox></object>
</annotation>"""
        files.save(files.join(ann_dir, "sample.xml"), xml_content, verbose=False)

        zip_target = files.join(self.root, "dataset.zip")
        out = datasets.convert(source=voc_dir, target=zip_target, src_format="voc")
        self.assertIsNone(out)
        self.assertTrue(files.is_file(zip_target))

        # Check with datasets.info
        info = datasets.info(zip_target)
        self.assertEqual(info["total_images"], 1)
        self.assertEqual(info["total_labels"], 1)


if __name__ == "__main__":
    unittest.main()
