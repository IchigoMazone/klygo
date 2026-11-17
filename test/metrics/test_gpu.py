"""Real CuPy execution parity for every metric and grouped storage operation."""
import unittest
from unittest.mock import patch
import numpy as np
from klygo import metrics, outputs, datasets, files, media
from tempfile import TemporaryDirectory
from PIL import Image

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    cp = None
    GPU_AVAILABLE = False


def pair(seed=0, packed=False):
    rng = np.random.default_rng(seed)
    predictions, targets = [], []
    for index, count in enumerate((0, 7, 15)):
        xy = rng.integers(0, 100, size=(count, 2))
        wh = rng.integers(5, 110, size=(count, 2))
        boxes = np.column_stack((xy, xy+wh)).tolist()
        gt = [outputs.Box(box, label=('cat', 'dog')[i % 2]) for i, box in enumerate(boxes)]
        pred = [outputs.Box(box, label=('cat', 'dog')[(i+int(i % 5 == 0)) % 2], score=(i % 4)/4)
                for i, box in enumerate(boxes)]
        if count:
            pred += [outputs.Box(boxes[1], label='dog', score=.95),
                     outputs.Box((200,200,210,210), label='ghost', score=.5)]
        kwargs = dict(source_id='image', frame_index=index, image_size=(300, 300))
        predictions.append(outputs.Detection(pred, **kwargs))
        targets.append(outputs.Detection(gt, **kwargs))
    return outputs.Detections(predictions, packed=packed), outputs.Detections(targets[::-1], packed=packed)


def assert_nested(case, actual, expected):
    if isinstance(expected, dict):
        case.assertEqual(set(actual), set(expected))
        for key in expected: assert_nested(case, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(type(actual), type(expected))
        case.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected): assert_nested(case, a, b)
    elif isinstance(expected, (float, int)):
        case.assertAlmostEqual(actual, expected, places=11)
    else:
        case.assertEqual(actual, expected)


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA device are required')
class TestGPUMetrics(unittest.TestCase):
    def setUp(self):
        self.names = [name for name in metrics.__all__ if name not in ('compose', 'evaluate')]
        self.evaluator = metrics.compose(*(getattr(metrics, name)() for name in self.names))

    def test_all_metrics_direct_composed_single_and_collection(self):
        for packed in (False, True):
            for dtype in (16, 32, 64):
                pred, target = pair(packed=packed)
                group = outputs.DetectionGroup(pred, target).astype(dtype)
                expected = metrics.evaluate(pred, target, self.evaluator)
                group.cuda(0)
                before = pred.to_dict(), target.to_dict()
                # Metrics must never unpack individual Box objects to obtain pixels/coordinates.
                with patch.object(outputs.Detection, '__iter__', side_effect=AssertionError('Box iteration in metrics')):
                    actual = metrics.evaluate(pred, target, self.evaluator)
                    for name in self.names:
                        assert_nested(self, getattr(metrics, name)(pred, target), actual[name])
                    single = metrics.evaluate(pred[1], target[1], self.evaluator)
                assert_nested(self, actual, expected)
                self.assertEqual(before, (pred.to_dict(), target.to_dict()))
                self.assertEqual((pred.device, target.device), ('cuda:0', 'cuda:0'))
                group.cpu()
                single_cpu = metrics.evaluate(pred[1], target[1], self.evaluator)
                assert_nested(self, single, single_cpu)

    def test_empty_collections_and_empty_sides(self):
        fixtures = [(outputs.Detections(), outputs.Detections()),
                    (outputs.Detections([outputs.Detection([], source_id='x', image_size=(20,20))]),
                     outputs.Detections([outputs.Detection([], source_id='x', image_size=(20,20))]))]
        for pred, target in fixtures:
            expected = metrics.evaluate(pred, target, self.evaluator)
            outputs.DetectionGroup(pred, target).cuda(0)
            assert_nested(self, metrics.evaluate(pred, target, self.evaluator), expected)

    def test_dataset_group_load_cuda_metrics(self):
        with TemporaryDirectory() as directory:
            root = files.path(directory)
            for name in ("images", "labels", "pseudo_labels", "third_labels"):
                files.mkdir(root / name)
            files.save(root / "data.yaml", {"names": {0: "cat"}}, overwrite=True, verbose=False)
            media.save(root / "images" / "a.png", Image.new("RGB", (100, 100)), verbose=False)
            for name in ("labels", "pseudo_labels", "third_labels"):
                files.save(root / name / "a.txt", "0 0.5 0.5 0.2 0.2\n", overwrite=True, verbose=False)
            sets = datasets.load(root, labels_dir=["pseudo_labels", "labels", "third_labels"], verbose=False)
            expected = metrics.evaluate(sets[0], sets[1], self.evaluator)
            self.assertEqual(list(sets.keys()), ["set_1", "set_2", "set_3"])
            sets.cuda(0).astype(32).pack()
            self.assertTrue(all(member.device == "cuda:0" for member in sets))
            members = sets.members
            sets.rename({"set_1": "pred", "set_2": "truth"})
            self.assertEqual(sets.members, members)
            self.assertTrue(all(member.device == "cuda:0" for member in sets))
            assert_nested(self, metrics.evaluate(sets["pred"], sets["truth"], self.evaluator), expected)
            sets.keep("cat").relabel({"cat": "animal"}).compact().unpack().cpu()
            self.assertTrue(all(member[0].labels == ["animal"] for member in sets))

    def test_mixed_devices_rejected_without_implicit_conversion(self):
        pred, target = pair()
        pred.cuda(0)
        with self.assertRaises(ValueError): metrics.evaluate(pred, target)
        self.assertEqual(target.device, 'cpu')
        outputs.DetectionGroup(pred, target).cpu()

    def test_grouped_gpu_operations_and_array_ownership(self):
        pred, target = pair()
        group = outputs.DetectionGroup(pred, target).cuda(0).astype(32).pack()
        self.assertIs(group.cuda(0), group)
        self.assertIsInstance(pred[1].to_arrays()['boxes'], cp.ndarray)
        arrays = pred[1].to_arrays()
        before = pred[1].boxes
        arrays['boxes'].fill(0)
        self.assertEqual(pred[1].boxes, before)
        group.keep(['cat', 'dog']).drop(['dog']).relabel({'cat': 'animal'}).compact().unpack()
        self.assertEqual(pred.device, 'cuda:0')
        self.assertTrue(all(set(frame.labels) <= {'animal'} for frame in pred))
        group.cpu().astype(64)
        self.assertEqual((pred.device, target.device, pred.dtype, target.dtype), ('cpu','cpu',64,64))

    def test_randomized_cpu_gpu_parity(self):
        for seed in range(5):
            pred, target = pair(seed)
            expected = metrics.evaluate(pred, target, self.evaluator)
            outputs.DetectionGroup(pred, target).cuda(0)
            assert_nested(self, metrics.evaluate(pred, target, self.evaluator), expected)


if __name__ == '__main__': unittest.main()
