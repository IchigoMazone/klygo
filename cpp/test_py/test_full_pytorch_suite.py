import klygo
import unittest

class TestFullPyTorchSuite(unittest.TestCase):
    def test_manual_seed(self):
        klygo.manual_seed(42)
        r1 = klygo.rand(2, 3)
        klygo.manual_seed(42)
        r2 = klygo.rand(2, 3)
        self.assertEqual(r1.tolist(), r2.tolist())

    def test_cat_stack_split_chunk_where(self):
        a = klygo.ones(2, 3)
        b = klygo.zeros(2, 3)
        
        # cat
        c = klygo.cat([a, b], dim=0)
        self.assertEqual(c.shape, (4, 3))
        
        # stack
        s = klygo.stack([a, b], dim=0)
        self.assertEqual(s.shape, (2, 2, 3))
        
        # split & chunk
        sp = klygo.split(c, 2, dim=0)
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].shape, (2, 3))
        
        ch = klygo.chunk(c, 2, dim=0)
        self.assertEqual(len(ch), 2)

        # where
        cond = klygo.tensor([[True, False, True], [False, True, False]])
        w = klygo.where(cond, a, b)
        self.assertEqual(w.shape, (2, 3))
        self.assertEqual(w[0, 0].item(), 1.0)
        self.assertEqual(w[0, 1].item(), 0.0)

    def test_indexing_slicing(self):
        x = klygo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(x[1, 2].item(), 6.0)
        self.assertEqual(x[0:2].shape, (2, 3))

        x[0, 0] = 99.0
        self.assertEqual(x[0, 0].item(), 99.0)

    def test_shape_and_memory_ops(self):
        x = klygo.zeros(2, 3)
        self.assertEqual(x.T.shape, (3, 2))
        self.assertTrue(x.is_contiguous)
        self.assertEqual(x.element_size(), 4)

        u = x.unsqueeze(0)
        self.assertEqual(u.shape, (1, 2, 3))
        sq = u.squeeze(0)
        self.assertEqual(sq.shape, (2, 3))

        p = x.permute(1, 0)
        self.assertEqual(p.shape, (3, 2))

        fl = x.flatten()
        self.assertEqual(fl.shape, (6,))

        cl = x.clone()
        self.assertEqual(cl.shape, x.shape)

    def test_unary_math_and_clamp(self):
        x = klygo.tensor([1.0, 4.0, 9.0])
        self.assertEqual(x.sqrt().tolist(), [1.0, 2.0, 3.0])
        self.assertEqual((-x).tolist(), [-1.0, -4.0, -9.0])
        self.assertEqual((x ** 2).tolist(), [1.0, 16.0, 81.0])

        c = klygo.tensor([-5.0, 0.0, 10.0]).clamp(0.0, 5.0)
        self.assertEqual(c.tolist(), [0.0, 0.0, 5.0])

    def test_inplace_ops(self):
        x = klygo.ones(2, 2)
        x += 2.0
        self.assertEqual(x.tolist(), [[3.0, 3.0], [3.0, 3.0]])
        x.fill_(7.0)
        self.assertEqual(x.tolist(), [[7.0, 7.0], [7.0, 7.0]])
        x.zero_()
        self.assertEqual(x.tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_comparisons(self):
        a = klygo.tensor([1.0, 2.0, 3.0])
        b = klygo.tensor([2.0, 2.0, 2.0])
        
        eq = (a == b)
        self.assertEqual(eq.tolist(), [0.0, 1.0, 0.0]) # or bool
        gt = (a > b)
        self.assertEqual(gt.tolist(), [0.0, 0.0, 1.0])

    def test_reductions_and_stats(self):
        x = klygo.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.max().item(), 4.0)
        self.assertEqual(x.min().item(), 1.0)
        self.assertEqual(x.prod().item(), 24.0)
        self.assertAlmostEqual(x.mean().item(), 2.5)
        self.assertTrue(x.all().item())
        self.assertTrue(x.any().item())

if __name__ == "__main__":
    unittest.main()
