from types import SimpleNamespace
import unittest
from unittest.mock import patch

from thinker.submission.crypto import (
    pack_signed_adapter_bundle,
    unpack_signed_adapter_bundle,
)


class SubmissionCryptoTests(unittest.TestCase):
    def test_signed_bundle_is_bound_to_miner_epoch_and_adapter_hash(self) -> None:
        files = {
            "adapter_config.json": b'{"peft_type":"LORA"}',
            "adapter_model.safetensors": b"weights",
        }
        raw_signature = bytes.fromhex("11" * 64)
        wallet = SimpleNamespace(
            hotkey=SimpleNamespace(
                sign=lambda payload: raw_signature.hex().encode("ascii"),
            )
        )
        payload = pack_signed_adapter_bundle(
            files,
            wallet=wallet,
            netuid=16,
            epoch=123,
            miner_hotkey="5miner",
        )

        def fake_verify(miner_hotkey, manifest, signature):
            self.assertEqual(signature, raw_signature)
            return True

        with patch(
            "thinker.submission.crypto.verify_submission_manifest_signature",
            side_effect=fake_verify,
        ):
            self.assertEqual(
                unpack_signed_adapter_bundle(
                    payload,
                    expected_miner_hotkey="5miner",
                    expected_epoch=123,
                    expected_netuid=16,
                ),
                files,
            )
            with self.assertRaisesRegex(ValueError, "miner mismatch"):
                unpack_signed_adapter_bundle(
                    payload,
                    expected_miner_hotkey="5copy",
                    expected_epoch=123,
                    expected_netuid=16,
                )
            with self.assertRaisesRegex(ValueError, "epoch mismatch"):
                unpack_signed_adapter_bundle(
                    payload,
                    expected_miner_hotkey="5miner",
                    expected_epoch=124,
                    expected_netuid=16,
                )


if __name__ == "__main__":
    unittest.main()
