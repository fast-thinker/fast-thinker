from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from thinker.submission.crypto import (
    EncryptedSubmission,
    decrypt_as_recipient,
    encrypt_for_recipients,
    submission_from_json,
    submission_to_json,
)

COMMON_SAMPLE_DIVISOR = 2
COMMON_SEED_PATH_PREFIX = "common-seeds"
_COMMON_SEED_VERSION = 1
_MAX_COMMON_SEED_FILE_BYTES = 512 * 1024
_SEED_RE = re.compile(r"[0-9a-f]{64}")
_REPO_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
)


class CommonSeedError(ValueError):
    pass


@dataclass(frozen=True)
class CommonSeedRecord:
    owner_hotkey: str
    seed: str


@dataclass(frozen=True)
class SampleSeedPlan:
    seeds: tuple[str, ...]
    common_count: int


def _validate_epoch(epoch: int) -> None:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CommonSeedError("epoch must be a non-negative integer")


def _validate_seed(seed: str) -> None:
    if not isinstance(seed, str) or _SEED_RE.fullmatch(seed) is None:
        raise CommonSeedError("common seed must be 32-byte lowercase hex")


def common_seed_commitment(seed: str) -> str:
    _validate_seed(seed)
    return hashlib.sha256(
        b"thinker-owner-common-seed-v1\0" + bytes.fromhex(seed)
    ).hexdigest()


def common_seed_filename(seed_commitment_hex: str) -> str:
    if not isinstance(seed_commitment_hex, str) or _SEED_RE.fullmatch(seed_commitment_hex) is None:
        raise CommonSeedError("seed commitment must be 32-byte lowercase hex")
    return f"{COMMON_SEED_PATH_PREFIX}/{seed_commitment_hex}.json"


def encrypt_common_seed(
    record: CommonSeedRecord,
    recipient_pubkeys: dict[str, bytes],
) -> EncryptedSubmission:
    _validate_seed(record.seed)
    if not isinstance(record.owner_hotkey, str) or not record.owner_hotkey:
        raise CommonSeedError("owner hotkey is required")
    if not recipient_pubkeys:
        raise CommonSeedError("no validator encryption public keys were found")
    payload = json.dumps(
        {
            "owner_hotkey": record.owner_hotkey,
            "seed": record.seed,
            "v": _COMMON_SEED_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encrypt_for_recipients(payload, recipient_pubkeys)


def decrypt_common_seed(
    submission: EncryptedSubmission,
    *,
    recipient_id: str,
    recipient_privkey: bytes,
    expected_owner_hotkey: str,
    expected_seed_commitment: str,
) -> CommonSeedRecord:
    common_seed_filename(expected_seed_commitment)
    try:
        plaintext = decrypt_as_recipient(submission, recipient_id, recipient_privkey)
        data = json.loads(plaintext)
    except Exception as exc:
        raise CommonSeedError("common seed could not be decrypted") from exc
    if not isinstance(data, dict) or set(data) != {"owner_hotkey", "seed", "v"}:
        raise CommonSeedError("common seed payload has unexpected fields")
    if data.get("v") != _COMMON_SEED_VERSION:
        raise CommonSeedError("unsupported common seed version")
    if data.get("owner_hotkey") != expected_owner_hotkey:
        raise CommonSeedError("common seed is from a different owner")
    seed = data.get("seed")
    _validate_seed(seed)
    if common_seed_commitment(seed) != expected_seed_commitment:
        raise CommonSeedError("common seed does not match the owner chain commitment")
    return CommonSeedRecord(expected_owner_hotkey, seed)


def upload_common_seed(
    repo_id: str,
    record: CommonSeedRecord,
    submission: EncryptedSubmission,
    *,
    token: str,
    api: Any | None = None,
) -> str:
    if not isinstance(repo_id, str) or _REPO_RE.fullmatch(repo_id) is None:
        raise CommonSeedError("invalid Hugging Face repo id")
    if not isinstance(token, str) or not token:
        raise CommonSeedError("a Hugging Face write token is required")
    seed_commitment = common_seed_commitment(record.seed)
    filename = common_seed_filename(seed_commitment)
    raw = submission_to_json(submission).encode("utf-8")
    if len(raw) > _MAX_COMMON_SEED_FILE_BYTES:
        raise CommonSeedError("encrypted common seed bundle is too large")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=raw,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Set Thinker persistent common seed {seed_commitment[:12]}",
        token=token,
    )
    return filename


def fetch_common_seed(
    repo_id: str,
    *,
    seed_commitment_hex: str,
    recipient_id: str,
    recipient_privkey: bytes,
    owner_hotkey: str,
    token: str | None = None,
    cache_dir: str | None = None,
    download_fn: Callable[..., str] | None = None,
    max_recipients: int = 256,
) -> CommonSeedRecord:
    if not isinstance(repo_id, str) or _REPO_RE.fullmatch(repo_id) is None:
        raise CommonSeedError("invalid Hugging Face repo id")
    filename = common_seed_filename(seed_commitment_hex)
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download
    try:
        local_path = download_fn(
            repo_id=repo_id,
            filename=filename,
            revision=None,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
        path = Path(local_path)
        if not path.is_file() or path.stat().st_size > _MAX_COMMON_SEED_FILE_BYTES:
            raise CommonSeedError("common seed bundle is missing or oversized")
        raw = path.read_text(encoding="utf-8")
        submission = submission_from_json(
            raw,
            max_ciphertext_bytes=512,
            max_recipients=max_recipients,
        )
    except CommonSeedError:
        raise
    except Exception as exc:
        raise CommonSeedError("common seed bundle could not be fetched") from exc
    return decrypt_common_seed(
        submission,
        recipient_id=recipient_id,
        recipient_privkey=recipient_privkey,
        expected_owner_hotkey=owner_hotkey,
        expected_seed_commitment=seed_commitment_hex,
    )


def _derived_seed(root: str, *, epoch: int, namespace: str, index: int) -> str:
    material = (
        b"thinker-common-sample-v1\0"
        + root.encode("ascii")
        + b"\0"
        + str(epoch).encode("ascii")
        + b"\0"
        + namespace.encode("utf-8")
        + b"\0"
        + str(index).encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def build_sample_seed_plan(
    count: int,
    *,
    private_seed: str,
    epoch: int,
    namespace: str,
    common_seed: str | None,
) -> SampleSeedPlan:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CommonSeedError("sample count must be a non-negative integer")
    _validate_epoch(epoch)
    if not isinstance(private_seed, str) or not private_seed:
        raise CommonSeedError("private seed is required")
    if not isinstance(namespace, str) or not namespace:
        raise CommonSeedError("sample namespace is required")
    common_count = 0
    if common_seed is not None:
        _validate_seed(common_seed)
        common_count = count // COMMON_SAMPLE_DIVISOR
    common = tuple(
        _derived_seed(common_seed, epoch=epoch, namespace=namespace, index=index)
        for index in range(common_count)
    )
    private = tuple(
        _derived_seed(private_seed, epoch=epoch, namespace=namespace, index=index)
        for index in range(count - common_count)
    )
    return SampleSeedPlan(common + private, common_count)


__all__ = [
    "COMMON_SAMPLE_DIVISOR",
    "COMMON_SEED_PATH_PREFIX",
    "CommonSeedError",
    "CommonSeedRecord",
    "SampleSeedPlan",
    "build_sample_seed_plan",
    "common_seed_commitment",
    "common_seed_filename",
    "decrypt_common_seed",
    "encrypt_common_seed",
    "fetch_common_seed",
    "upload_common_seed",
]
