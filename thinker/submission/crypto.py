from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_DATA_KEY_BITS = 256
_NONCE_LEN = 12
_HKDF_INFO = b"thinker-lora-submission-v1"
_MAX_DEFAULT_ADAPTER_BYTES = 200 * 1024 * 1024
_MAX_DEFAULT_CONFIG_BYTES = 64 * 1024
_MAX_DEFAULT_RECIPIENTS = 256
_MAX_RECIPIENT_ID_BYTES = 256
_ALLOWED_ADAPTER_FILES = frozenset({"adapter_config.json", "adapter_model.safetensors"})


class SubmissionFormatError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SubmissionFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except SubmissionFormatError:
        raise
    except json.JSONDecodeError as exc:
        raise SubmissionFormatError(f"invalid JSON: {exc}") from exc


def _max_b64_chars(max_decoded_bytes: int) -> int:
    return 4 * ((max_decoded_bytes + 2) // 3)


def max_encrypted_adapter_ciphertext_bytes(max_adapter_bytes: int) -> int:
    return _max_b64_chars(max_adapter_bytes) + 4096 + 16


def max_submission_json_bytes(
    max_adapter_bytes: int, max_recipients: int = _MAX_DEFAULT_RECIPIENTS
) -> int:
    ciphertext_bytes = max_encrypted_adapter_ciphertext_bytes(max_adapter_bytes)
    return _max_b64_chars(ciphertext_bytes) + max_recipients * 1024 + 4096


def _decode_b64(value: Any, *, field: str, max_bytes: int) -> bytes:
    if not isinstance(value, str) or len(value) > _max_b64_chars(max_bytes):
        raise SubmissionFormatError(f"{field} is missing or exceeds its size limit")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise SubmissionFormatError(f"{field} is not valid base64") from exc
    if len(decoded) > max_bytes:
        raise SubmissionFormatError(f"{field} exceeds its size limit")
    return decoded


def generate_keypair() -> tuple[bytes, bytes]:
    priv = X25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return priv_bytes, pub_bytes


def public_key_from_private(privkey_bytes: bytes) -> bytes:
    priv = X25519PrivateKey.from_private_bytes(privkey_bytes)
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


@dataclass(frozen=True)
class WrappedKey:
    ephemeral_pubkey: bytes
    nonce: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class EncryptedSubmission:
    nonce: bytes
    ciphertext: bytes
    wrapped_keys: dict[str, WrappedKey]


def _derive_symmetric_key(shared_material: bytes, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=_DATA_KEY_BITS // 8, salt=salt, info=_HKDF_INFO
    ).derive(shared_material)


def wrap_key(
    data_key: bytes, recipient_pubkey_bytes: bytes, *, nonce: bytes | None = None
) -> WrappedKey:
    ephemeral_priv = X25519PrivateKey.generate()
    ephemeral_pub_bytes = ephemeral_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    recipient_pub = X25519PublicKey.from_public_bytes(recipient_pubkey_bytes)
    shared_material = ephemeral_priv.exchange(recipient_pub)
    symmetric_key = _derive_symmetric_key(shared_material, salt=ephemeral_pub_bytes)
    if nonce is None:
        nonce = os.urandom(_NONCE_LEN)
    ciphertext = AESGCM(symmetric_key).encrypt(nonce, data_key, None)
    return WrappedKey(ephemeral_pubkey=ephemeral_pub_bytes, nonce=nonce, ciphertext=ciphertext)


def unwrap_key(wrapped: WrappedKey, recipient_privkey_bytes: bytes) -> bytes:
    recipient_priv = X25519PrivateKey.from_private_bytes(recipient_privkey_bytes)
    ephemeral_pub = X25519PublicKey.from_public_bytes(wrapped.ephemeral_pubkey)
    shared_material = recipient_priv.exchange(ephemeral_pub)
    symmetric_key = _derive_symmetric_key(shared_material, salt=wrapped.ephemeral_pubkey)
    return AESGCM(symmetric_key).decrypt(wrapped.nonce, wrapped.ciphertext, None)


def encrypt_for_recipients(
    plaintext: bytes, recipient_pubkeys: dict[str, bytes]
) -> EncryptedSubmission:
    data_key = AESGCM.generate_key(bit_length=_DATA_KEY_BITS)
    nonce = os.urandom(_NONCE_LEN)
    ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, None)
    wrapped_keys = {
        recipient_id: wrap_key(data_key, pubkey) for recipient_id, pubkey in recipient_pubkeys.items()
    }
    return EncryptedSubmission(nonce=nonce, ciphertext=ciphertext, wrapped_keys=wrapped_keys)


def decrypt_as_recipient(
    submission: EncryptedSubmission, recipient_id: str, recipient_privkey_bytes: bytes
) -> bytes:
    if recipient_id not in submission.wrapped_keys:
        raise KeyError(f"{recipient_id!r} is not a recipient of this submission")
    data_key = unwrap_key(submission.wrapped_keys[recipient_id], recipient_privkey_bytes)
    return AESGCM(data_key).decrypt(submission.nonce, submission.ciphertext, None)


_VALIDATOR_KEY_GRANT_NONCE = bytes(_NONCE_LEN)


def encrypt_validator_key_for_owner(
    validator_privkey_bytes: bytes, owner_pubkey_bytes: bytes
) -> WrappedKey:
    if len(validator_privkey_bytes) != 32:
        raise SubmissionFormatError("validator private key must be 32 raw bytes")
    # A fixed (all-zero) nonce is safe here, unlike the general AEAD case: wrap_key
    # generates a fresh, never-reused ephemeral X25519 key per call, so the derived
    # AES-GCM key is unique to this one encryption regardless of nonce -- the same
    # sealed-box construction libsodium's crypto_box_seal uses. This lets the nonce
    # be omitted from the on-chain commitment (chain commitments cap at 128 raw
    # bytes; the 12-byte nonce would otherwise push the bundle over that limit).
    return wrap_key(validator_privkey_bytes, owner_pubkey_bytes, nonce=_VALIDATOR_KEY_GRANT_NONCE)


def decrypt_validator_key_as_owner(
    wrapped: WrappedKey, owner_privkey_bytes: bytes
) -> bytes:
    validator_privkey_bytes = unwrap_key(wrapped, owner_privkey_bytes)
    if len(validator_privkey_bytes) != 32:
        raise SubmissionFormatError("decrypted validator private key has invalid length")
    return validator_privkey_bytes


def validator_key_grant_owner_key_id(owner_pubkey_bytes: bytes) -> str:
    if len(owner_pubkey_bytes) != 32:
        raise SubmissionFormatError("owner public key must be 32 raw bytes")
    return hashlib.sha256(
        b"thinker-owner-x25519-v1\0" + owner_pubkey_bytes
    ).digest()[:16].hex()


def validator_key_grant_fields(wrapped: WrappedKey) -> dict[str, str]:
    # nonce_hex is deliberately not included -- see _VALIDATOR_KEY_GRANT_NONCE above.
    return {
        "ephemeral_pubkey_hex": wrapped.ephemeral_pubkey.hex(),
        "ciphertext_hex": wrapped.ciphertext.hex(),
    }


def validator_key_grant_from_fields(
    ephemeral_pubkey_hex: str, ciphertext_hex: str
) -> WrappedKey:
    return WrappedKey(
        ephemeral_pubkey=bytes.fromhex(ephemeral_pubkey_hex),
        nonce=_VALIDATOR_KEY_GRANT_NONCE,
        ciphertext=bytes.fromhex(ciphertext_hex),
    )


def content_hash(submission: EncryptedSubmission) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"thinker-encrypted-submission-v1\0")

    def _field(value: bytes) -> None:
        hasher.update(len(value).to_bytes(8, "big"))
        hasher.update(value)

    _field(submission.nonce)
    _field(submission.ciphertext)
    for recipient_id in sorted(submission.wrapped_keys):
        wrapped = submission.wrapped_keys[recipient_id]
        _field(recipient_id.encode("utf-8"))
        _field(wrapped.ephemeral_pubkey)
        _field(wrapped.nonce)
        _field(wrapped.ciphertext)
    return hasher.hexdigest()


def pack_adapter_bundle(
    files: dict[str, bytes],
    *,
    max_total_bytes: int = _MAX_DEFAULT_ADAPTER_BYTES,
    max_config_bytes: int = _MAX_DEFAULT_CONFIG_BYTES,
) -> bytes:
    if not isinstance(files, dict) or set(files) != _ALLOWED_ADAPTER_FILES:
        raise SubmissionFormatError(
            "adapter bundle must contain exactly adapter_config.json and adapter_model.safetensors"
        )
    if any(not isinstance(name, str) or not isinstance(data, bytes) for name, data in files.items()):
        raise SubmissionFormatError("adapter bundle names must be strings and contents must be bytes")
    if not files["adapter_model.safetensors"]:
        raise SubmissionFormatError("adapter_model.safetensors is empty")
    if not files["adapter_config.json"] or len(files["adapter_config.json"]) > max_config_bytes:
        raise SubmissionFormatError("adapter_config.json is empty or exceeds its size limit")
    if sum(len(data) for data in files.values()) > max_total_bytes:
        raise SubmissionFormatError("adapter bundle exceeds its total size limit")
    return json.dumps(
        {"files": {name: base64.b64encode(data).decode("ascii") for name, data in files.items()}}
    ).encode("utf-8")


def unpack_adapter_bundle(
    plaintext: bytes,
    *,
    max_total_bytes: int = _MAX_DEFAULT_ADAPTER_BYTES,
    max_config_bytes: int = _MAX_DEFAULT_CONFIG_BYTES,
) -> dict[str, bytes]:
    max_plaintext_bytes = _max_b64_chars(max_total_bytes) + 4096
    if not isinstance(plaintext, bytes) or len(plaintext) > max_plaintext_bytes:
        raise SubmissionFormatError("decrypted adapter bundle exceeds its encoded size limit")
    try:
        data = _strict_json_loads(plaintext.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SubmissionFormatError("adapter bundle is not UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != {"files"} or not isinstance(data["files"], dict):
        raise SubmissionFormatError("adapter bundle must contain only a files object")
    if set(data["files"]) != _ALLOWED_ADAPTER_FILES:
        raise SubmissionFormatError(
            "adapter bundle must contain exactly adapter_config.json and adapter_model.safetensors"
        )
    files = {
        "adapter_config.json": _decode_b64(
            data["files"]["adapter_config.json"],
            field="adapter_config.json",
            max_bytes=max_config_bytes,
        ),
        "adapter_model.safetensors": _decode_b64(
            data["files"]["adapter_model.safetensors"],
            field="adapter_model.safetensors",
            max_bytes=max_total_bytes,
        ),
    }
    if not files["adapter_config.json"] or not files["adapter_model.safetensors"]:
        raise SubmissionFormatError("adapter files must not be empty")
    if sum(len(value) for value in files.values()) > max_total_bytes:
        raise SubmissionFormatError("decoded adapter bundle exceeds its total size limit")
    return files


def submission_to_json(submission: EncryptedSubmission) -> str:
    def _b64(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return json.dumps(
        {
            "nonce": _b64(submission.nonce),
            "ciphertext": _b64(submission.ciphertext),
            "wrapped_keys": {
                rid: {
                    "ephemeral_pubkey": _b64(w.ephemeral_pubkey),
                    "nonce": _b64(w.nonce),
                    "ciphertext": _b64(w.ciphertext),
                }
                for rid, w in submission.wrapped_keys.items()
            },
        }
    )


def submission_from_json(
    raw: str,
    *,
    max_ciphertext_bytes: int | None = None,
    max_recipients: int = _MAX_DEFAULT_RECIPIENTS,
) -> EncryptedSubmission:
    if max_ciphertext_bytes is None:
        max_ciphertext_bytes = _max_b64_chars(_MAX_DEFAULT_ADAPTER_BYTES) + 4096 + 16
    max_json_bytes = _max_b64_chars(max_ciphertext_bytes) + max_recipients * 1024 + 4096
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > max_json_bytes:
        raise SubmissionFormatError("submission JSON exceeds its size limit")
    data = _strict_json_loads(raw)
    if not isinstance(data, dict) or set(data) != {"nonce", "ciphertext", "wrapped_keys"}:
        raise SubmissionFormatError("submission JSON has unexpected fields")
    wrapped_raw = data["wrapped_keys"]
    if not isinstance(wrapped_raw, dict) or not 1 <= len(wrapped_raw) <= max_recipients:
        raise SubmissionFormatError("wrapped_keys has an invalid recipient count")

    wrapped_keys: dict[str, WrappedKey] = {}
    for rid, value in wrapped_raw.items():
        if (
            not isinstance(rid, str)
            or not rid
            or len(rid.encode("utf-8")) > _MAX_RECIPIENT_ID_BYTES
            or any(ord(ch) < 32 for ch in rid)
        ):
            raise SubmissionFormatError("invalid recipient id")
        if not isinstance(value, dict) or set(value) != {"ephemeral_pubkey", "nonce", "ciphertext"}:
            raise SubmissionFormatError(f"invalid wrapped key for recipient {rid!r}")
        ephemeral_pubkey = _decode_b64(
            value["ephemeral_pubkey"], field=f"{rid}.ephemeral_pubkey", max_bytes=32
        )
        nonce = _decode_b64(value["nonce"], field=f"{rid}.nonce", max_bytes=_NONCE_LEN)
        wrapped_ciphertext = _decode_b64(
            value["ciphertext"], field=f"{rid}.ciphertext", max_bytes=32 + 16
        )
        if len(ephemeral_pubkey) != 32 or len(nonce) != _NONCE_LEN or len(wrapped_ciphertext) != 48:
            raise SubmissionFormatError(f"invalid cryptographic field length for recipient {rid!r}")
        wrapped_keys[rid] = WrappedKey(ephemeral_pubkey, nonce, wrapped_ciphertext)

    nonce = _decode_b64(data["nonce"], field="nonce", max_bytes=_NONCE_LEN)
    ciphertext = _decode_b64(
        data["ciphertext"], field="ciphertext", max_bytes=max_ciphertext_bytes
    )
    if len(nonce) != _NONCE_LEN or len(ciphertext) < 16:
        raise SubmissionFormatError("invalid encrypted payload lengths")
    return EncryptedSubmission(nonce=nonce, ciphertext=ciphertext, wrapped_keys=wrapped_keys)
