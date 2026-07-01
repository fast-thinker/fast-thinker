from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from thinker.config import ThinkerConfig
from thinker.submission.adapter_validation import ValidatedAdapter, validate_adapter_files


_PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40,64}")


@contextmanager
def materialize_validated_adapter(
    adapter_files: Mapping[str, bytes], policy: ThinkerConfig
) -> Iterator[tuple[str, ValidatedAdapter]]:
    validated = validate_adapter_files(adapter_files, policy)
    with tempfile.TemporaryDirectory(prefix="thinker_lora_") as tmpdir:
        root = Path(tmpdir)
        config_path = root / "adapter_config.json"
        weights_path = root / "adapter_model.safetensors"
        config_path.write_bytes(validated.files["adapter_config.json"])
        weights_path.write_bytes(validated.files["adapter_model.safetensors"])
        yield tmpdir, validated


def load_peft_adapter(
    base_model: Any,
    adapter_files: Mapping[str, bytes],
    policy: ThinkerConfig,
    *,
    adapter_name: str = "default",
) -> Any:
    from peft import LoraConfig, PeftModel

    with materialize_validated_adapter(adapter_files, policy) as (adapter_dir, validated):
        lora_config = LoraConfig(**validated.config)
        return PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            adapter_name=adapter_name,
            is_trainable=False,
            config=lora_config,
            local_files_only=True,
            use_safetensors=True,
        )


def load_frozen_base_model(
    repo_id: str,
    revision: str,
    *,
    token: str | None = None,
    **model_kwargs: Any,
) -> Any:
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("base-model repo_id is required")
    if not isinstance(revision, str) or _PINNED_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("base-model revision must be an immutable 40-64 character commit hash")
    forbidden = {"trust_remote_code", "use_safetensors", "revision", "token"} & set(model_kwargs)
    if forbidden:
        raise ValueError(f"security-sensitive model kwargs cannot be overridden: {sorted(forbidden)}")

    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        repo_id,
        revision=revision,
        token=token,
        trust_remote_code=False,
        use_safetensors=True,
        **model_kwargs,
    )
