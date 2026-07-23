import klygo
import unittest

class TestParallelAndDevice(unittest.TestCase):
    def test_device_properties(self):
        x = klygo.zeros([2, 3])
        self.assertEqual(x.device, klygo.device("cpu"))
        self.assertEqual(str(x.device), "device(type='cpu')")

    def test_device_to_cpu(self):
        x = klygo.ones([4, 4])
        y = x.to("cpu")
        self.assertEqual(y.device, klygo.device("cpu"))
        self.assertEqual(y.shape, (4, 4))
        self.assertEqual(y.ndim, 2)

    def test_device_to_cuda_raises(self):
        x = klygo.ones([2, 2])
        # Since CUDA is not compiled in this CPU build, it should throw a runtime_error
        with self.assertRaises(Exception) as context:
            x.to("cuda")
        self.assertIn("CUDA is not enabled", str(context.exception))

    def test_openmp_parallel_math(self):
        # Create larger tensors to exercise OpenMP multi-threading
        a = klygo.ones([1000, 1000])
        b = klygo.full([1000, 1000], 2.0)
        c = a + b
        self.assertEqual(c.shape, (1000, 1000))
        self.assertEqual(c.sum().shape, ())
        self.assertEqual(c.sum().dtype, klygo.float32)

if __name__ == "__main__":
    unittest.main()
