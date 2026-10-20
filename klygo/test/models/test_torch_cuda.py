"""Ensure model CUDA behavior delegates directly to PyTorch."""

import importlib.util
import unittest
from unittest.mock import patch

import torch

import klygo
from klygo.models.backend import common, torch_runtime


class TestTorchCudaDelegation(unittest.TestCase):
    def test_cuda_is_not_a_top_level_klygo_module(self):
        self.assertNotIn("cuda", klygo.__all__)
        self.assertFalse(hasattr(klygo, "cuda"))
        self.assertIsNone(importlib.util.find_spec("klygo.cuda"))
        self.assertFalse(hasattr(klygo.utils, "is_cuda_available"))
        self.assertFalse(hasattr(klygo.utils, "get_gpu_name"))

    def test_synchronize_uses_torch_directly(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True) as available,
            patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            torch_runtime.synchronize()
        available.assert_called_once_with()
        synchronize.assert_called_once_with()

    def test_synchronize_does_nothing_on_cpu(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            torch_runtime.synchronize()
        synchronize.assert_not_called()

    def test_clear_cache_uses_torch_directly(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            common.clear_cache("Hugging Face")
        empty_cache.assert_called_once_with()

    def test_autocast_device_defaults_from_torch(self):
        sentinel = object()
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.amp, "autocast", return_value=sentinel) as autocast,
        ):
            context = torch_runtime.autocast(use_half=True)
        self.assertIs(context, sentinel)
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()
