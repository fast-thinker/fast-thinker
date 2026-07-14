from types import SimpleNamespace
import unittest

from thinker.validator.cli import (
    _accepted_result_count,
    _ensure_enc_pubkey_published,
    _run_mode_tags,
    build_parser,
)


_HOTKEY = "5owner"
_PUBKEY_HEX = "11" * 32
_SEED_COMMITMENT_HEX = "22" * 32


class _FakeSubtensor:
    def __init__(self, *, pubkey_hex: str | None = _PUBKEY_HEX) -> None:
        self.pubkey_hex = pubkey_hex
        self.commits = 0

    def metagraph(self, _netuid: int):
        return SimpleNamespace(hotkeys=[_HOTKEY], validator_permit=[True])

    def get_commitment_metadata(self, _netuid: int, _hotkey: str):
        if self.pubkey_hex is None:
            return None
        raw = (
            b"TKM1"
            + bytes.fromhex(self.pubkey_hex)
            + bytes.fromhex(_SEED_COMMITMENT_HEX)
        )
        return {"info": {"fields": [{"Raw68": raw}]}}

    def set_commitment(self, *_args, **_kwargs):
        self.commits += 1
        return True, ""


def _wallet(hotkey: str = _HOTKEY):
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=hotkey))


class ValidatorCliTests(unittest.TestCase):
    def test_run_accepts_no_wandb_with_no_set_weights(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--wallet",
                "validator-wallet",
                "--hotkey",
                "validator-hotkey",
                "--evaluation-delay-epochs",
                "6",
                "--burn-rate",
                "1",
                "--no-set-weights",
                "--no-wandb",
            ]
        )

        self.assertTrue(args.no_set_weights)
        self.assertTrue(args.no_wandb)
        self.assertIsNone(args.test_mode)
        self.assertEqual(
            _run_mode_tags(args, args.test_mode),
            ("no_set_weights", "no_wandb"),
        )

    def test_test_mode_tags_disable_weights_and_wandb(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--test-mode", "math"])

        self.assertEqual(
            _run_mode_tags(args, args.test_mode),
            ("test_mode", "no_set_weights", "no_wandb"),
        )

    def test_accepted_result_count_ignores_rejected_results(self) -> None:
        results = {
            "accepted": SimpleNamespace(score=object()),
            "rejected": SimpleNamespace(score=None),
        }

        self.assertEqual(_accepted_result_count(results), 1)

    def test_accepted_result_count_zero_when_all_rejected(self) -> None:
        results = {
            "a": SimpleNamespace(score=None),
            "b": SimpleNamespace(score=None),
        }

        self.assertEqual(_accepted_result_count(results), 0)

    def test_owner_hotkey_can_confirm_as_validator_from_owner_common_seed(self) -> None:
        subtensor = _FakeSubtensor()

        _ensure_enc_pubkey_published(
            subtensor,
            _wallet(),
            16,
            b"\x00" * 32,
            bytes.fromhex(_PUBKEY_HEX),
            owner_hotkey=_HOTKEY,
        )

        self.assertEqual(subtensor.commits, 0)

    def test_owner_validator_requires_owner_common_seed_commitment(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "owner common-seed commitment"):
            _ensure_enc_pubkey_published(
                _FakeSubtensor(pubkey_hex=None),
                _wallet(),
                16,
                b"\x00" * 32,
                bytes.fromhex(_PUBKEY_HEX),
                owner_hotkey=_HOTKEY,
            )

    def test_owner_validator_requires_matching_encryption_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _ensure_enc_pubkey_published(
                _FakeSubtensor(pubkey_hex="33" * 32),
                _wallet(),
                16,
                b"\x00" * 32,
                bytes.fromhex(_PUBKEY_HEX),
                owner_hotkey=_HOTKEY,
            )


if __name__ == "__main__":
    unittest.main()
