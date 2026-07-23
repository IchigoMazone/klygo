import klygo
import unittest

class TestNewFactoryFunctions(unittest.TestCase):
    def test_linspace(self):
        l = klygo.linspace(0.0, 1.0, 5)
        self.assertEqual(l.shape, (5,))
        self.assertEqual(l.dtype, klygo.float32)

    def test_rand_randn(self):
        r = klygo.rand(2, 3)
        self.assertEqual(r.shape, (2, 3))
        self.assertEqual(r.dtype, klygo.float32)

        rn = klygo.randn(2, 3, dtype=klygo.float64)
        self.assertEqual(rn.shape, (2, 3))
        self.assertEqual(rn.dtype, klygo.float64)

    def test_randint(self):
        ri1 = klygo.randint(0, 10, (2, 3))
        self.assertEqual(ri1.shape, (2, 3))

        ri2 = klygo.randint(10, (2, 3))
        self.assertEqual(ri2.shape, (2, 3))

    def test_like_functions(self):
        x = klygo.ones(3, 4, dtype=klygo.int32)
        
        el = klygo.empty_like(x)
        self.assertEqual(el.shape, (3, 4))
        self.assertEqual(el.dtype, klygo.int32)

        zl = klygo.zeros_like(x)
        self.assertEqual(zl.shape, (3, 4))
        self.assertEqual(zl.dtype, klygo.int32)

        ol = klygo.ones_like(x, dtype=klygo.float64)
        self.assertEqual(ol.shape, (3, 4))
        self.assertEqual(ol.dtype, klygo.float64)

        fl = klygo.full_like(x, 9.0)
        self.assertEqual(fl.shape, (3, 4))

    def test_tensor_from_data(self):
        t1 = klygo.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t1.shape, (2, 2))
        self.assertEqual(t1.dtype, klygo.float32)

        t2 = klygo.tensor(42)
        self.assertEqual(t2.shape, ())

if __name__ == "__main__":
    unittest.main()
