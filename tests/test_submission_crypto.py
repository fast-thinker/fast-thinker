import base64
import json
import unittest

from thinker.submission.crypto import (
    pack_bound_adapter_bundle,
    unpack_bound_adapter_bundle,
)


class SubmissionCryptoTests(unittest.TestCase):
    def test_bound_bundle_is_bound_to_miner_epoch_and_adapter_hash(self) -> None:
        files = {
            "adapter_config.json": b'{"peft_type":"LORA"}',
            "adapter_model.safetensors": b"weights",
        }
        payload = pack_bound_adapter_bundle(
            files,
            netuid=16,
            epoch=123,
            miner_hotkey="5miner",
        )

        self.assertEqual(
            unpack_bound_adapter_bundle(
                payload,
                expected_miner_hotkey="5miner",
                expected_epoch=123,
                expected_netuid=16,
            ),
            files,
        )
        with self.assertRaisesRegex(ValueError, "miner mismatch"):
            unpack_bound_adapter_bundle(
                payload,
                expected_miner_hotkey="5copy",
                expected_epoch=123,
                expected_netuid=16,
            )
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            unpack_bound_adapter_bundle(
                payload,
                expected_miner_hotkey="5miner",
                expected_epoch=124,
                expected_netuid=16,
            )

    def test_bound_bundle_ignores_legacy_signature_field(self) -> None:
        files = {
            "adapter_config.json": b'{"peft_type":"LORA"}',
            "adapter_model.safetensors": b"weights",
        }
        payload = pack_bound_adapter_bundle(
            files,
            netuid=16,
            epoch=123,
            miner_hotkey="5miner",
        )
        data = json.loads(payload.decode("utf-8"))
        data["signature"] = base64.b64encode(b"legacy").decode("ascii")
        legacy_payload = json.dumps(data, separators=(",", ":")).encode("utf-8")

        self.assertEqual(
            unpack_bound_adapter_bundle(
                legacy_payload,
                expected_miner_hotkey="5miner",
                expected_epoch=123,
                expected_netuid=16,
            ),
            files,
        )


if __name__ == "__main__":
    unittest.main()
