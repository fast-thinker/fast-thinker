from __future__ import annotations

import importlib.metadata
import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=object))
sys.modules.setdefault("safetensors", types.ModuleType("safetensors"))
_safetensors_torch = types.ModuleType("safetensors.torch")
_safetensors_torch.load = lambda *args, **kwargs: {}
sys.modules.setdefault("safetensors.torch", _safetensors_torch)

from thinker.validator.inference import _validate_flashinfer_jit_cache


def _metadata_versions(versions: dict[str, str]):
    def version(distribution: str) -> str:
        if distribution not in versions:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    return version


class FlashInferEnvironmentTest(unittest.TestCase):
    def test_allows_missing_jit_cache(self) -> None:
        with patch(
            "thinker.validator.inference.importlib.metadata.version",
            _metadata_versions({"flashinfer-python": "0.6.11.post2"}),
        ):
            _validate_flashinfer_jit_cache()

    def test_allows_matching_public_jit_cache_version(self) -> None:
        with patch(
            "thinker.validator.inference.importlib.metadata.version",
            _metadata_versions(
                {
                    "flashinfer-python": "0.6.11.post2",
                    "flashinfer-jit-cache": "0.6.11.post2+cu130",
                }
            ),
        ):
            _validate_flashinfer_jit_cache()

    def test_rejects_stale_jit_cache_version(self) -> None:
        with patch(
            "thinker.validator.inference.importlib.metadata.version",
            _metadata_versions(
                {
                    "flashinfer-python": "0.6.11.post2",
                    "flashinfer-jit-cache": "0.6.8.post1+cu130",
                }
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "uv pip uninstall flashinfer-jit-cache"):
                _validate_flashinfer_jit_cache()


if __name__ == "__main__":
    unittest.main()
