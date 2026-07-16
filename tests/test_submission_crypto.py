import base64
import json
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

    def test_unpack_normalizes_hex_encoded_signature_bytes(self) -> None:
        files = {
            "adapter_config.json": b'{"peft_type":"LORA"}',
            "adapter_model.safetensors": b"weights",
        }
        raw_signature = bytes.fromhex("22" * 64)
        wallet = SimpleNamespace(
            hotkey=SimpleNamespace(
                sign=lambda payload: raw_signature,
            )
        )
        payload = pack_signed_adapter_bundle(
            files,
            wallet=wallet,
            netuid=16,
            epoch=123,
            miner_hotkey="5miner",
        )
        data = json.loads(payload.decode("utf-8"))
        data["signature"] = base64.b64encode(
            raw_signature.hex().encode("ascii")
        ).decode("ascii")
        hex_encoded_payload = json.dumps(data, separators=(",", ":")).encode("utf-8")

        def fake_verify(miner_hotkey, manifest, signature):
            self.assertEqual(signature, raw_signature)
            return True

        with patch(
            "thinker.submission.crypto.verify_submission_manifest_signature",
            side_effect=fake_verify,
        ):
            self.assertEqual(
                unpack_signed_adapter_bundle(
                    hex_encoded_payload,
                    expected_miner_hotkey="5miner",
                    expected_epoch=123,
                    expected_netuid=16,
                ),
                files,
            )


if __name__ == "__main__":
    unittest.main()
