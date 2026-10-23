"""Torchvision backend hooks without selecting a model architecture."""

import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "install klygo[torchvision]")
class TestTorchvisionBackend(unittest.TestCase):
    def test_runtime_hooks_accept_a_caller_owned_module(self):
        import torch
        from klygo.models.backend import torchvision

        class CallerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                return [
                    {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                     "scores": torch.tensor([0.9]),
                     "labels": torch.tensor([7])}
                    for _ in images
                ]

        model = CallerModel().eval()
        inputs = torchvision.cast_inputs(
            [torch.zeros((3, 8, 8))],
            dev=torch.device("cpu"),
            dtype=torch.float32,
        )
        results = torchvision.format_results(torchvision.run_inference(model, inputs))
        self.assertEqual(results[0]["labels"], [7])
        self.assertEqual(results[0]["class_ids"], [7])
        self.assertEqual(results[0]["boxes"], [[1.0, 2.0, 3.0, 4.0]])

        with tempfile.TemporaryDirectory() as temporary:
            torchvision.save(model, temporary)
            from klygo import files
            self.assertTrue(files.is_file(files.join(temporary, "model.pt")))


if __name__ == "__main__":
    unittest.main()
