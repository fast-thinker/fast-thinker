from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from safetensors.torch import load as load_safetensors

from thinker.config import ThinkerConfig, validate_adapter_bounds


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ALLOWED_ADAPTER_FILES = frozenset({ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME})

_KNOWN_CONFIG_KEYS = frozenset(
    {
        "task_type",
        "peft_type",
        "auto_mapping",
        "peft_version",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
        "r",
        "target_modules",
        "exclude_modules",
        "lora_alpha",
        "lora_dropout",
        "fan_in_fan_out",
        "bias",
        "use_rslora",
        "modules_to_save",
        "init_lora_weights",
        "layers_to_transform",
        "layers_pattern",
        "rank_pattern",
        "alpha_pattern",
        "megatron_config",
        "megatron_core",
        "trainable_token_indices",
        "loftq_config",
        "eva_config",
        "corda_config",
        "lora_ga_config",
        "use_dora",
        "alora_invocation_tokens",
        "use_qalora",
        "qalora_group_size",
        "layer_replication",
        "lora_bias",
        "target_parameters",
        "use_bdlora",
        "arrow_config",
        "ensure_weight_tying",
    }
)
_INERT_FIELDS: dict[str, tuple[Any, ...]] = {
    "auto_mapping": (None,),
    "exclude_modules": (None,),
    "modules_to_save": (None, ()),
    "layers_to_transform": (None,),
    "layers_pattern": (None,),
    "rank_pattern": (None, ()),
    "alpha_pattern": (None, ()),
    "megatron_config": (None,),
    "trainable_token_indices": (None,),
    "loftq_config": (None, ()),
    "eva_config": (None,),
    "corda_config": (None,),
    "lora_ga_config": (None,),
    "alora_invocation_tokens": (None,),
    "layer_replication": (None,),
    "target_parameters": (None,),
    "use_bdlora": (None, False),
    "arrow_config": (None,),
}
_TENSOR_NAME_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<module>[A-Za-z0-9_]+)\.lora_(?P<side>A|B)(?:\.default)?\.weight$"
)


class AdapterValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedAdapter:
    files: dict[str, bytes]
    config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdapterValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_empty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and not value


def _is_empty_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not value


def _parse_adapter_config_json(raw: bytes, policy: ThinkerConfig) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise AdapterValidationError("adapter_config.json must be bytes")
    if not raw or len(raw) > policy.max_adapter_config_bytes:
        raise AdapterValidationError("adapter_config.json is empty or exceeds its size limit")
    try:
        config = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AdapterValidationError(f"non-finite JSON number: {value}")
            ),
        )
    except AdapterValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterValidationError(f"invalid adapter_config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise AdapterValidationError("adapter_config.json must contain a JSON object")
    return config


def _validate_known_config_fields(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - _KNOWN_CONFIG_KEYS)
    if unknown:
        raise AdapterValidationError(f"unsupported adapter config fields: {unknown}")


def _validate_lora_identity_and_targets(
    config: Mapping[str, Any],
    policy: ThinkerConfig,
) -> list[str]:
    ok, reason = validate_adapter_bounds(
        policy,
        rank=config.get("r"),
        target_modules=config.get("target_modules"),
        n_bytes=0,
    )
    if not ok:
        raise AdapterValidationError(reason)

    target_modules = config["target_modules"]
    if len(set(target_modules)) != len(target_modules):
        raise AdapterValidationError("target_modules contains duplicates")
    if config.get("peft_type", "LORA") != "LORA":
        raise AdapterValidationError("only PEFT LORA adapters are supported")
    if config.get("task_type") not in (None, "CAUSAL_LM"):
        raise AdapterValidationError("only CAUSAL_LM LoRA adapters are supported")
    if config.get("bias", "none") != "none":
        raise AdapterValidationError("LoRA bias weights are not supported")
    return list(target_modules)


def _validate_inert_config_fields(config: Mapping[str, Any]) -> None:
    for field, allowed in _INERT_FIELDS.items():
        if field not in config:
            continue
        value = config[field]
        allowed_value = value in allowed
        if () in allowed and (_is_empty_mapping(value) or _is_empty_sequence(value)):
            allowed_value = True
        if not allowed_value:
            raise AdapterValidationError(f"adapter config field {field!r} must be disabled")


def _validate_lora_feature_flags(config: Mapping[str, Any]) -> None:
    for field in ("use_dora", "use_qalora", "lora_bias", "ensure_weight_tying"):
        value = config.get(field, False)
        if not isinstance(value, bool) or value:
            raise AdapterValidationError(f"adapter feature {field!r} is not supported")
    for field in ("fan_in_fan_out", "use_rslora", "inference_mode"):
        if field in config and not isinstance(config[field], bool):
            raise AdapterValidationError(f"adapter config field {field!r} must be boolean")


def _validate_lora_numeric_fields(
    config: Mapping[str, Any],
    policy: ThinkerConfig,
) -> None:
    alpha = config.get("lora_alpha", config["r"])
    dropout = config.get("lora_dropout", 0.0)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha <= policy.max_lora_alpha
    ):
        raise AdapterValidationError("lora_alpha must be a positive finite number")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0.0 <= dropout <= 1.0
    ):
        raise AdapterValidationError("lora_dropout must be between 0 and 1")


def _validate_base_model_fields(
    config: Mapping[str, Any],
    policy: ThinkerConfig,
) -> None:
    configured_base = config.get("base_model_name_or_path")
    if configured_base is not None and (
        not isinstance(configured_base, str) or len(configured_base) > 512
    ):
        raise AdapterValidationError("base_model_name_or_path must be a bounded string")
    if policy.base_model_repo and configured_base not in (None, "", policy.base_model_repo):
        raise AdapterValidationError("adapter targets a different base model")
    configured_revision = config.get("revision")
    if configured_revision is not None and (
        not isinstance(configured_revision, str) or len(configured_revision) > 128
    ):
        raise AdapterValidationError("revision must be a bounded string")
    if configured_revision not in (None, "", policy.base_model_revision):
        raise AdapterValidationError("adapter targets a different base-model revision")
    peft_version = config.get("peft_version")
    if peft_version is not None and (
        not isinstance(peft_version, str) or len(peft_version) > 64
    ):
        raise AdapterValidationError("peft_version must be a bounded string")


def _normalized_adapter_config(
    config: Mapping[str, Any],
    policy: ThinkerConfig,
    target_modules: list[str],
) -> dict[str, Any]:
    normalized = dict(config)
    normalized["peft_type"] = "LORA"
    normalized["task_type"] = "CAUSAL_LM"
    normalized["inference_mode"] = True
    normalized["init_lora_weights"] = False
    normalized["megatron_core"] = "megatron.core"
    normalized["qalora_group_size"] = 16
    normalized["target_modules"] = sorted(target_modules)
    if policy.base_model_repo:
        normalized["base_model_name_or_path"] = policy.base_model_repo
        normalized["revision"] = policy.base_model_revision
    return normalized


def parse_and_validate_adapter_config(raw: bytes, policy: ThinkerConfig) -> dict[str, Any]:
    config = _parse_adapter_config_json(raw, policy)
    _validate_known_config_fields(config)
    target_modules = _validate_lora_identity_and_targets(config, policy)
    _validate_inert_config_fields(config)
    _validate_lora_feature_flags(config)
    _validate_lora_numeric_fields(config, policy)
    _validate_base_model_fields(config, policy)
    return _normalized_adapter_config(config, policy, target_modules)


def _validate_state_dict(
    state_dict: dict[str, torch.Tensor], config: Mapping[str, Any], policy: ThinkerConfig
) -> None:
    if not state_dict:
        raise AdapterValidationError("safetensors file contains no tensors")
    if len(state_dict) > policy.max_adapter_tensors:
        raise AdapterValidationError("safetensors file contains too many tensors")

    pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    actual_modules: set[str] = set()
    for name, tensor in state_dict.items():
        if len(name) > 1024 or any(ord(ch) < 32 for ch in name):
            raise AdapterValidationError("invalid tensor name")
        match = _TENSOR_NAME_RE.fullmatch(name)
        if match is None:
            raise AdapterValidationError(f"unsupported tensor name: {name}")
        module = match.group("module")
        if module not in config["target_modules"]:
            raise AdapterValidationError(f"tensor targets undeclared module: {module}")
        if not tensor.dtype.is_floating_point:
            raise AdapterValidationError(f"tensor {name!r} must use a floating-point dtype")
        if tensor.ndim != 2 or tensor.numel() == 0:
            raise AdapterValidationError(f"tensor {name!r} must be a non-empty matrix")
        if not bool(torch.isfinite(tensor).all().item()):
            raise AdapterValidationError(f"tensor {name!r} contains NaN or infinity")
        if float(tensor.detach().to(torch.float32).abs().max().item()) > policy.max_abs_adapter_weight:
            raise AdapterValidationError(f"tensor {name!r} exceeds the absolute weight limit")
        pair_key = (match.group("prefix"), module)
        side = match.group("side")
        if side in pairs.setdefault(pair_key, {}):
            raise AdapterValidationError(f"duplicate LoRA {side} tensor for {pair_key}")
        pairs[pair_key][side] = tensor
        actual_modules.add(module)

    rank = config["r"]
    for pair_key, sides in pairs.items():
        if set(sides) != {"A", "B"}:
            raise AdapterValidationError(f"incomplete LoRA tensor pair: {pair_key}")
        if sides["A"].shape[0] != rank or sides["B"].shape[1] != rank:
            raise AdapterValidationError(f"LoRA tensor rank does not match config for {pair_key}")
    if actual_modules != set(config["target_modules"]):
        missing = sorted(set(config["target_modules"]) - actual_modules)
        raise AdapterValidationError(f"target_modules have no weights: {missing}")


def validate_adapter_files(
    files: Mapping[str, bytes], policy: ThinkerConfig
) -> ValidatedAdapter:
    if not isinstance(files, Mapping):
        raise AdapterValidationError("adapter bundle must be a file mapping")
    names = set(files)
    if names != ALLOWED_ADAPTER_FILES:
        extra = sorted(names - ALLOWED_ADAPTER_FILES)
        missing = sorted(ALLOWED_ADAPTER_FILES - names)
        raise AdapterValidationError(f"adapter files must be exact; extra={extra}, missing={missing}")
    if any(not isinstance(value, bytes) for value in files.values()):
        raise AdapterValidationError("adapter files must contain bytes")

    total_bytes = sum(len(value) for value in files.values())
    ok, reason = validate_adapter_bounds(
        policy,
        rank=1,
        target_modules=[policy.allowed_target_modules[0]],
        n_bytes=total_bytes,
    )
    if not ok:
        raise AdapterValidationError(reason)

    config = parse_and_validate_adapter_config(files[ADAPTER_CONFIG_NAME], policy)
    try:
        state_dict = load_safetensors(files[ADAPTER_WEIGHTS_NAME])
    except Exception as exc:
        raise AdapterValidationError(f"invalid safetensors weights: {exc}") from exc
    _validate_state_dict(state_dict, config, policy)

    canonical_files = {
        ADAPTER_CONFIG_NAME: json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ADAPTER_WEIGHTS_NAME: files[ADAPTER_WEIGHTS_NAME],
    }
    return ValidatedAdapter(files=canonical_files, config=config, state_dict=state_dict)
