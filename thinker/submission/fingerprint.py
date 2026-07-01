from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

_HASH_DOMAIN = b"thinker-lora-fingerprint-v2\0"
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass
class LoraFingerprint:
    exact_hash: str
    arch_hash: str
    layer_names_hash: str
    param_count: int

    def to_dict(self) -> dict:
        return {
            "exact_hash": self.exact_hash,
            "arch_hash": self.arch_hash,
            "layer_names_hash": self.layer_names_hash,
            "param_count": self.param_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoraFingerprint":
        return cls(
            exact_hash=d["exact_hash"],
            arch_hash=d["arch_hash"],
            layer_names_hash=d["layer_names_hash"],
            param_count=d["param_count"],
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_config_bytes(adapter_config: dict | None) -> bytes:
    return json.dumps(
        adapter_config or {},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _update_field(hasher, data: bytes) -> None:
    hasher.update(len(data).to_bytes(8, "big"))
    hasher.update(data)


def _update_tensor_bytes(hasher, tensor: torch.Tensor) -> None:
    raw = memoryview(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).cast("B")
    hasher.update(len(raw).to_bytes(8, "big"))
    for offset in range(0, len(raw), _HASH_CHUNK_BYTES):
        hasher.update(raw[offset : offset + _HASH_CHUNK_BYTES])


def _architecture_hash(
    state_dict: dict[str, torch.Tensor], adapter_config: dict | None
) -> str:
    architecture = {
        "config": adapter_config or {},
        "tensors": [
            {
                "dtype": str(state_dict[name].dtype),
                "name": name,
                "shape": list(state_dict[name].shape),
            }
            for name in sorted(state_dict)
        ],
    }
    return _sha256_bytes(
        json.dumps(
            architecture,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def compute_lora_fingerprint(
    state_dict: dict[str, torch.Tensor],
    adapter_config: dict | None = None,
) -> LoraFingerprint:
    with torch.no_grad():
        return _compute_impl(state_dict, adapter_config)


def _compute_impl(
    state_dict: dict[str, torch.Tensor],
    adapter_config: dict | None,
) -> LoraFingerprint:
    all_names = sorted(state_dict)
    total_params = sum(t.numel() for t in state_dict.values())

    layer_names_hash = _sha256_bytes("\n".join(all_names).encode())

    hasher = hashlib.sha256()
    hasher.update(_HASH_DOMAIN)
    _update_field(hasher, _canonical_config_bytes(adapter_config))
    for name in all_names:
        tensor = state_dict[name]
        _update_field(hasher, name.encode("utf-8"))
        _update_field(hasher, str(tensor.dtype).encode("ascii"))
        shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        _update_field(hasher, shape)
        _update_tensor_bytes(hasher, tensor)
    exact_hash = hasher.hexdigest()

    return LoraFingerprint(
        exact_hash=exact_hash,
        arch_hash=_architecture_hash(state_dict, adapter_config),
        layer_names_hash=layer_names_hash,
        param_count=total_params,
    )


def fingerprints_collide(
    new_fp: LoraFingerprint,
    existing_fp: LoraFingerprint,
) -> tuple[bool, str]:
    if new_fp.exact_hash == existing_fp.exact_hash:
        return True, "exact_adapter_copy"

    return False, ""
