"""Regression checks for conversion data preservation and validation."""
import copy
import tempfile
import unittest

from PIL import Image
from klygo import datasets, files, media


class ConversionValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = files.path(self.temp.name)
        self.source = self.root / 'source'
        self.images = self.source / 'images'
        files.mkdir(self.images)
        media.save(self.images / 'one.jpg', Image.new('RGB', (100, 100)))
        self.annotation = self.source / 'instances.json'
        self.target = self.root / 'output'
        self.data = {
            'images': [{'id': 1, 'file_name': 'one.jpg', 'width': 100, 'height': 100}],
            'categories': [{'id': 7, 'name': 'object'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 7, 'bbox': [10, 10, 20, 20]}],
        }

    def convert(self, target=None, **options):
        files.save(self.annotation, self.data, overwrite=True, verbose=False)
        return datasets.convert(images=self.images, annotations=self.annotation,
                                target=target or self.target, src_format='coco', **options)

    def test_overlapping_destination_preserves_source(self):
        for target in (self.source, self.images, self.images / 'nested'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'overlap'):
                self.convert(target, overwrite=True)
            self.assertTrue(files.is_file(self.images / 'one.jpg'))
            self.assertTrue(files.is_file(self.annotation))

    def test_symlink_overlap_preserves_source(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.convert(alias, overwrite=True)
        self.assertTrue(files.is_file(self.annotation))

    def test_invalid_boxes_do_not_replace_existing_output(self):
        files.mkdir(self.target)
        marker = self.target / 'keep.txt'
        files.save(marker, 'original', verbose=False)
        for bbox in ([float('nan'), 10, 20, 20], [10, float('inf'), 20, 20],
                     [10, 10, -1, 20], [10, 10, 0, 20], [10, 10, 20], [200, 10, 20, 20]):
            with self.subTest(bbox=bbox):
                self.data['annotations'][0]['bbox'] = bbox
                with self.assertRaises(ValueError):
                    self.convert(overwrite=True)
                self.assertEqual(files.load(marker, verbose=False), 'original')

    def test_missing_image_and_unknown_class_raise(self):
        original = copy.deepcopy(self.data)
        self.data['images'][0]['file_name'] = 'missing.jpg'
        with self.assertRaises(FileNotFoundError):
            self.convert()
        self.assertFalse(files.exists(self.target))
        self.data = original
        self.data['annotations'][0]['category_id'] = 999
        with self.assertRaisesRegex(ValueError, 'category'):
            self.convert()

    def test_invalid_dimensions_raise(self):
        for width in (float('nan'), float('inf'), 0, -1):
            self.data['images'][0]['width'] = width
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, 'dimensions'):
                self.convert()

    def test_success_returns_none_and_keeps_image_bytes(self):
        before = files.hash(self.images / 'one.jpg')
        self.assertIsNone(self.convert())
        self.assertEqual(files.hash(self.target / 'images' / 'one.jpg'), before)
        self.assertEqual(files.load(self.target / 'labels' / 'one.txt', as_lines=True, verbose=False),
                         ['0 0.200000 0.200000 0.200000 0.200000'])

    def test_unknown_options_raise(self):
        with self.assertRaisesRegex(TypeError, 'unexpected'):
            self.convert(invalid_option=True)

    def test_voc_nonfinite_box_and_missing_image_raise(self):
        voc = self.root / 'voc'
        images, annotations = voc / 'JPEGImages', voc / 'Annotations'
        files.mkdir(images)
        files.mkdir(annotations)
        media.save(images / 'one.jpg', Image.new('RGB', (100, 100)))
        files.save(annotations / 'one.xml', '<annotation><size><width>100</width><height>100</height></size>'
                   '<object><name>object</name><bndbox><xmin>nan</xmin><ymin>10</ymin>'
                   '<xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>', verbose=False)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            datasets.convert(voc, self.target)
        with self.assertRaises(FileNotFoundError):
            datasets.convert(voc, self.target, image_ids=['missing'])
