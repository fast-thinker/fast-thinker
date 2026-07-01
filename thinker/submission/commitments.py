from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

_COMMITMENT_KEY = "tk"
_COMMITMENT_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_X25519_PUBKEY_RE = re.compile(r"[0-9a-f]{64}")
_OWNER_KEY_ID_RE = re.compile(r"[0-9a-f]{32}")
_OWNER_COMMON_SEED_MAGIC = b"TKM1"
_OWNER_COMMON_SEED_BUNDLE_BYTES = 68
_WRAPPED_KEY_CIPHERTEXT_RE = re.compile(r"[0-9a-f]{96}")
# A Bittensor chain commitment caps out at 128 raw bytes no matter which
# encoding is used to publish it (set_commitment ultimately wraps whatever
# string/bytes it's given as Data::RawN, and the chain's Data enum only
# defines Raw0..Raw128 variants -- text JSON commits are bound by the exact
# same 128-byte ceiling as raw byte bundles, not a separate, larger one).
# The validator-keys bundle is published as those 128 raw bytes directly
# (see _commit_raw_bundle) rather than as JSON+base64 text, since
# base64-encoding even just this bundle's unavoidable cryptographic material
# (pubkey + ephemeral pubkey + ciphertext, 112 bytes minimum) already
# exceeds 128 *characters* before any JSON wrapper -- there is no text
# encoding that fits this much key material in the limit, only literal
# bytes do.
_VALIDATOR_KEYS_BUNDLE_BYTES = 128
# Hugging Face is the only submission transport this subnet supports, so the
# chain commitment carries repo_id + sha256. The filename is derived as
# f"{SUBMISSION_FILENAME_PREFIX}/{sha256}.json", and the revision is unpinned
# because validators re-hash every download before decrypting. This means the
# repo id is public chain metadata; miners who care about identity privacy
# should submit through a non-identifying Hugging Face account or organization.
# Keeping only repo_id + sha256 also fits the chain's 128-byte commitment cap.
_HF_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_SUBMISSION_BUNDLE_PREFIX = b"TKS1"
_SUBMISSION_BUNDLE_HEADER_BYTES = len(_SUBMISSION_BUNDLE_PREFIX) + 8 + 1
_SUBMISSION_BUNDLE_HASH_BYTES = 32
_SUBMISSION_BUNDLE_MAX_REPO_ID_BYTES = (
    128 - _SUBMISSION_BUNDLE_HEADER_BYTES - _SUBMISSION_BUNDLE_HASH_BYTES
)


class SubtensorLike(Protocol):
    def set_commitment(
        self, wallet: Any, netuid: int, data: str, **kwargs: Any
    ) -> Any: ...
    def get_commitment_metadata(self, netuid: int, hotkey: str) -> Optional[dict]: ...


@dataclass(frozen=True)
class EncPubkeyCommitment:
    pubkey_hex: str


@dataclass(frozen=True)
class SubmissionCommitment:
    epoch: int
    repo_id: str
    sha256: str
    # The miner controls ``epoch`` in the payload, so validators must use the
    # chain-authenticated commitment block when enforcing submission age.
    block: int | None = None


@dataclass(frozen=True)
class ValidatorKeyGrantCommitment:
    owner_key_id_hex: str
    ephemeral_pubkey_hex: str
    ciphertext_hex: str


@dataclass(frozen=True)
class OwnerCommonSeedCommitment:
    pubkey_hex: str
    seed_commitment_hex: str


def _interpret_commit_result(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple):
        return (
            bool(result[0]) if result else False,
            str(result[1]) if len(result) > 1 else str(result),
        )
    if isinstance(result, bool):
        return result, str(result)
    if hasattr(result, "success"):
        return bool(result.success), str(getattr(result, "message", result))
    if hasattr(result, "is_success"):
        return bool(result.is_success), str(getattr(result, "error_message", result))
    return False, f"unexpected set_commitment response: {result!r}"


def _commit(wallet: Any, subtensor: SubtensorLike, netuid: int, payload: dict) -> tuple[bool, str]:
    try:
        data = json.dumps(
            {_COMMITMENT_KEY: {**payload, "v": _COMMITMENT_VERSION}}, separators=(",", ":")
        )
        try:
            result = subtensor.set_commitment(
                wallet=wallet,
                netuid=netuid,
                data=data,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            result = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
        success, message = _interpret_commit_result(result)
        if not success:
            logger.warning("commitment rejected by chain: %s", message)
            return False, f"chain rejected: {message}"
        return True, ""
    except Exception as exc:
        logger.warning("failed to commit: %s", exc, exc_info=True)
        return False, str(exc)


def _commit_raw_bundle(
    wallet: Any, subtensor: SubtensorLike, netuid: int, raw_bytes: bytes
) -> tuple[bool, str]:
    """Publishes raw_bytes directly as a chain commitment (Raw<N> SCALE
    type), bypassing the JSON+base64 text path _commit uses -- see
    _VALIDATOR_KEYS_BUNDLE_BYTES for why text encoding can't carry this much
    key material within the chain's 128-byte commitment cap. An alternate
    implementation can provide
    `set_raw_commitment(wallet, netuid, data, **kwargs)` on its SubtensorLike
    to intercept this without needing the real bittensor SDK;
    production falls back to bittensor's own publish_metadata_extrinsic,
    which (unlike Subtensor.set_commitment) accepts already-raw bytes
    instead of a str it would re-encode."""
    try:
        set_raw = getattr(subtensor, "set_raw_commitment", None)
        if callable(set_raw):
            result = set_raw(
                wallet=wallet,
                netuid=netuid,
                data=raw_bytes,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        else:
            from bittensor.core.extrinsics.serving import publish_metadata_extrinsic

            result = publish_metadata_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                data_type=f"Raw{len(raw_bytes)}",
                data=raw_bytes,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        success, message = _interpret_commit_result(result)
        if not success:
            logger.warning("commitment rejected by chain: %s", message)
            return False, f"chain rejected: {message}"
        return True, ""
    except Exception as exc:
        logger.warning("failed to commit: %s", exc, exc_info=True)
        return False, str(exc)


def commit_enc_pubkey(
    wallet: Any, subtensor: SubtensorLike, netuid: int, pubkey_hex: str
) -> tuple[bool, str]:
    if not isinstance(pubkey_hex, str) or _X25519_PUBKEY_RE.fullmatch(pubkey_hex) is None:
        return False, "encryption public key must be 32-byte lowercase hex"
    return _commit(wallet, subtensor, netuid, {"t": "enc_pubkey", "pubkey": pubkey_hex})


def commit_validator_keys(
    wallet: Any,
    subtensor: SubtensorLike,
    netuid: int,
    *,
    pubkey_hex: str,
    owner_key_id_hex: str,
    ephemeral_pubkey_hex: str,
    ciphertext_hex: str,
) -> tuple[bool, str]:
    if _X25519_PUBKEY_RE.fullmatch(pubkey_hex) is None:
        return False, "validator public key must be 32-byte lowercase hex"
    if _OWNER_KEY_ID_RE.fullmatch(owner_key_id_hex) is None:
        return False, "owner key id must be 16-byte lowercase hex"
    if _X25519_PUBKEY_RE.fullmatch(ephemeral_pubkey_hex) is None:
        return False, "ephemeral public key must be 32-byte lowercase hex"
    if _WRAPPED_KEY_CIPHERTEXT_RE.fullmatch(ciphertext_hex) is None:
        return False, "ciphertext must be 48-byte lowercase hex"
    bundle = bytes.fromhex(
        pubkey_hex
        + owner_key_id_hex
        + ephemeral_pubkey_hex
        + ciphertext_hex
    )
    assert len(bundle) == _VALIDATOR_KEYS_BUNDLE_BYTES
    return _commit_raw_bundle(wallet, subtensor, netuid, bundle)


def commit_owner_common_seed(
    wallet: Any,
    subtensor: SubtensorLike,
    netuid: int,
    *,
    pubkey_hex: str,
    seed_commitment_hex: str,
) -> tuple[bool, str]:
    if _X25519_PUBKEY_RE.fullmatch(pubkey_hex) is None:
        return False, "owner public key must be 32-byte lowercase hex"
    if _SHA256_RE.fullmatch(seed_commitment_hex) is None:
        return False, "seed commitment must be 32-byte lowercase hex"
    bundle = (
        _OWNER_COMMON_SEED_MAGIC
        + bytes.fromhex(pubkey_hex)
        + bytes.fromhex(seed_commitment_hex)
    )
    assert len(bundle) == _OWNER_COMMON_SEED_BUNDLE_BYTES
    return _commit_raw_bundle(wallet, subtensor, netuid, bundle)


def commit_submission(
    wallet: Any,
    subtensor: SubtensorLike,
    netuid: int,
    epoch: int,
    repo_id: str,
    sha256: str,
) -> tuple[bool, str]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        return False, "epoch must be a non-negative integer"
    if epoch > (2**64 - 1):
        return False, "epoch is too large for chain commitment"
    if not isinstance(repo_id, str) or _HF_REPO_ID_RE.fullmatch(repo_id) is None:
        return False, "repo_id must be a valid Hugging Face owner/name repo id"
    repo_id_bytes = repo_id.encode("ascii")
    if len(repo_id_bytes) > _SUBMISSION_BUNDLE_MAX_REPO_ID_BYTES:
        return (
            False,
            f"repo_id must be at most {_SUBMISSION_BUNDLE_MAX_REPO_ID_BYTES} bytes "
            "to fit in a chain commitment",
        )
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        return False, "sha256 must be 64-character lowercase hex"
    raw = (
        _SUBMISSION_BUNDLE_PREFIX
        + epoch.to_bytes(8, "big")
        + bytes([len(repo_id_bytes)])
        + repo_id_bytes
        + bytes.fromhex(sha256)
    )
    return _commit_raw_bundle(wallet, subtensor, netuid, raw)


def _raw_bytes(value: Any) -> bytes:
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and not isinstance(value[0], int)
    ):
        value = value[0]
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        text = value[2:] if value.startswith("0x") else value
        return bytes.fromhex(text)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and 0 <= item <= 255 for item in value
    ):
        return bytes(value)
    raise TypeError(f"unsupported Raw commitment value: {type(value).__name__}")


def _read_raw_commitment(subtensor: SubtensorLike, netuid: int, hotkey: str) -> Optional[dict]:
    try:
        commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
        if not commit_data:
            return None
        fields = commit_data.get("info", {}).get("fields", [])
        if not fields or not fields[0] or isinstance(fields[0], str):
            return None
        raw_entry = fields[0]
        raw_key = next((k for k in raw_entry if k.startswith("Raw") and k[3:].isdigit()), None)
        if raw_key is None:
            return None
        raw_bytes = _raw_bytes(raw_entry[raw_key])
        if raw_bytes.startswith(_SUBMISSION_BUNDLE_PREFIX):
            inner = _decode_submission_bundle(raw_bytes)
            if inner is not None:
                inner["_chain_block"] = commit_data.get("block")
                return inner
        if len(raw_bytes) == _VALIDATOR_KEYS_BUNDLE_BYTES:
            # The validator-keys bundle is published as literal raw bytes
            # (see _commit_raw_bundle), not JSON -- no other commitment type
            # this module writes is ever exactly this length.
            return {"t": "validator_keys", "v": _COMMITMENT_VERSION, "raw_bundle": raw_bytes}
        if (
            len(raw_bytes) == _OWNER_COMMON_SEED_BUNDLE_BYTES
            and raw_bytes.startswith(_OWNER_COMMON_SEED_MAGIC)
        ):
            return {
                "t": "owner_common_seed",
                "v": _COMMITMENT_VERSION,
                "raw_bundle": raw_bytes,
            }
        payload = json.loads(raw_bytes)
        inner = payload.get(_COMMITMENT_KEY)
        if not isinstance(inner, dict) or inner.get("v") != _COMMITMENT_VERSION:
            return None
        return inner
    except Exception as exc:
        logger.warning(
            "could not read commitment for hotkey %s: %s: %s",
            hotkey[:8],
            type(exc).__name__,
            exc,
        )
        return None


def read_enc_pubkey(
    subtensor: SubtensorLike, netuid: int, hotkey: str
) -> Optional[EncPubkeyCommitment]:
    inner = _read_raw_commitment(subtensor, netuid, hotkey)
    if inner is None:
        return None
    if inner.get("t") == "enc_pubkey":
        pubkey_hex = inner.get("pubkey")
    elif inner.get("t") == "validator_keys":
        decoded = _decode_validator_keys_bundle(inner)
        if decoded is None:
            return None
        pubkey_hex = decoded[0]
    elif inner.get("t") == "owner_common_seed":
        raw = inner.get("raw_bundle")
        if not isinstance(raw, bytes) or len(raw) != _OWNER_COMMON_SEED_BUNDLE_BYTES:
            return None
        pubkey_hex = raw[4:36].hex()
    else:
        return None
    if not isinstance(pubkey_hex, str) or _X25519_PUBKEY_RE.fullmatch(pubkey_hex) is None:
        return None
    return EncPubkeyCommitment(pubkey_hex=pubkey_hex)


def read_owner_common_seed(
    subtensor: SubtensorLike, netuid: int, hotkey: str
) -> Optional[OwnerCommonSeedCommitment]:
    inner = _read_raw_commitment(subtensor, netuid, hotkey)
    if inner is None or inner.get("t") != "owner_common_seed":
        return None
    raw = inner.get("raw_bundle")
    if not isinstance(raw, bytes) or len(raw) != _OWNER_COMMON_SEED_BUNDLE_BYTES:
        return None
    return OwnerCommonSeedCommitment(
        pubkey_hex=raw[4:36].hex(),
        seed_commitment_hex=raw[36:68].hex(),
    )


def _decode_validator_keys_bundle(
    inner: dict[str, Any],
) -> tuple[str, ValidatorKeyGrantCommitment] | None:
    raw = inner.get("raw_bundle")
    if not isinstance(raw, bytes) or len(raw) != _VALIDATOR_KEYS_BUNDLE_BYTES:
        return None
    return (
        raw[:32].hex(),
        ValidatorKeyGrantCommitment(
            owner_key_id_hex=raw[32:48].hex(),
            ephemeral_pubkey_hex=raw[48:80].hex(),
            ciphertext_hex=raw[80:128].hex(),
        ),
    )


def _decode_submission_bundle(raw: bytes) -> dict[str, Any] | None:
    min_len = _SUBMISSION_BUNDLE_HEADER_BYTES + _SUBMISSION_BUNDLE_HASH_BYTES
    if len(raw) < min_len or len(raw) > 128 or not raw.startswith(_SUBMISSION_BUNDLE_PREFIX):
        return None
    epoch = int.from_bytes(
        raw[len(_SUBMISSION_BUNDLE_PREFIX):len(_SUBMISSION_BUNDLE_PREFIX) + 8], "big"
    )
    repo_id_len = raw[_SUBMISSION_BUNDLE_HEADER_BYTES - 1]
    expected_len = _SUBMISSION_BUNDLE_HEADER_BYTES + repo_id_len + _SUBMISSION_BUNDLE_HASH_BYTES
    if expected_len != len(raw) or repo_id_len < 3:
        return None
    repo_id_start = _SUBMISSION_BUNDLE_HEADER_BYTES
    repo_id_end = repo_id_start + repo_id_len
    try:
        repo_id = raw[repo_id_start:repo_id_end].decode("ascii")
    except UnicodeDecodeError:
        return None
    sha256 = raw[repo_id_end:].hex()
    if _HF_REPO_ID_RE.fullmatch(repo_id) is None:
        return None
    return {
        "t": "submission",
        "v": _COMMITMENT_VERSION,
        "epoch": epoch,
        "repo_id": repo_id,
        "sha256": sha256,
    }


def read_validator_key_grant(
    subtensor: SubtensorLike, netuid: int, hotkey: str
) -> Optional[ValidatorKeyGrantCommitment]:
    inner = _read_raw_commitment(subtensor, netuid, hotkey)
    if inner is None or inner.get("t") != "validator_keys":
        return None
    decoded = _decode_validator_keys_bundle(inner)
    return decoded[1] if decoded is not None else None


def read_all_validator_key_grants(
    subtensor: SubtensorLike, netuid: int, metagraph: Any
) -> dict[str, ValidatorKeyGrantCommitment]:
    result: dict[str, ValidatorKeyGrantCommitment] = {}
    for hotkey in metagraph.hotkeys:
        if not hotkey:
            continue
        grant = read_validator_key_grant(subtensor, netuid, hotkey)
        if grant is not None:
            result[hotkey] = grant
    return result


def read_submission(
    subtensor: SubtensorLike, netuid: int, hotkey: str, epoch: int
) -> Optional[SubmissionCommitment]:
    """Read the miner's latest submission as of ``epoch``.

    A miner commitment remains active until the miner replaces it.  The
    committed epoch is therefore a not-before value, not an expiry.  Future
    commitments are still ignored so a miner cannot enter an evaluation
    before the epoch for which it submitted.
    """
    inner = _read_raw_commitment(subtensor, netuid, hotkey)
    if inner is None or inner.get("t") != "submission":
        return None
    committed_epoch = inner.get("epoch")
    if (
        isinstance(committed_epoch, bool)
        or not isinstance(committed_epoch, int)
        or committed_epoch < 0
        or committed_epoch > epoch
    ):
        return None
    repo_id, sha256 = inner.get("repo_id"), inner.get("sha256")
    if (
        not isinstance(repo_id, str)
        or _HF_REPO_ID_RE.fullmatch(repo_id) is None
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
    ):
        return None
    return SubmissionCommitment(
        epoch=committed_epoch,
        repo_id=repo_id,
        sha256=sha256,
        block=(
            inner.get("_chain_block")
            if isinstance(inner.get("_chain_block"), int)
            and not isinstance(inner.get("_chain_block"), bool)
            and inner.get("_chain_block") >= 0
            else None
        ),
    )


def read_all_enc_pubkeys(subtensor: SubtensorLike, netuid: int, metagraph: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for hotkey in metagraph.hotkeys:
        if not hotkey:
            continue
        commitment = read_enc_pubkey(subtensor, netuid, hotkey)
        if commitment is not None:
            result[hotkey] = commitment.pubkey_hex
    return result
