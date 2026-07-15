from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


NETUID = 16


@dataclass(frozen=True)
class ThinkerConfig:
    base_model_repo: str = field(default_factory=lambda: _env_str("THINKER_BASE_MODEL_REPO", "Qwen/Qwen3.5-4B"))
    base_model_revision: str = field(
        default_factory=lambda: _env_str(
            "THINKER_BASE_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )
    )

    owner_hotkey: str = field(
        default_factory=lambda: _env_str(
            "THINKER_OWNER_HOTKEY", "5ECWmM21HNE834eB296ZDR8aBMW5psWcD5eJhFzdowKyrbNW"
        )
    )
    common_seed_repo: str = field(
        default_factory=lambda: _env_str("THINKER_COMMON_SEED_REPO", "")
    )

    max_lora_rank: int = field(default_factory=lambda: _env_int("THINKER_MAX_LORA_RANK", 128))
    max_adapter_bytes: int = field(
        default_factory=lambda: _env_int("THINKER_MAX_ADAPTER_BYTES", 500 * 1024 * 1024)
    )
    max_adapter_config_bytes: int = field(
        default_factory=lambda: _env_int("THINKER_MAX_ADAPTER_CONFIG_BYTES", 64 * 1024)
    )
    max_adapter_tensors: int = field(
        default_factory=lambda: _env_int("THINKER_MAX_ADAPTER_TENSORS", 4096)
    )
    max_lora_alpha: float = field(
        default_factory=lambda: _env_float("THINKER_MAX_LORA_ALPHA", 4096.0)
    )
    max_abs_adapter_weight: float = field(
        default_factory=lambda: _env_float("THINKER_MAX_ABS_ADAPTER_WEIGHT", 10_000.0)
    )
    max_submission_recipients: int = field(
        default_factory=lambda: _env_int("THINKER_MAX_SUBMISSION_RECIPIENTS", 256)
    )
    allowed_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    )

    n_problems_per_epoch: int = field(default_factory=lambda: _env_int("THINKER_N_PROBLEMS_PER_EPOCH", 50))
    n_long_context_qa_per_epoch: int = field(default_factory=lambda: _env_int("THINKER_N_LONG_CONTEXT_QA_PER_EPOCH", 0))
    qualification_math_per_epoch: int = field(default_factory=lambda: _env_int("THINKER_QUALIFICATION_MATH_PER_EPOCH", 0))
    qualification_long_context_qa_per_epoch: int = field(default_factory=lambda: _env_int("THINKER_QUALIFICATION_LONG_CONTEXT_QA_PER_EPOCH", 0))
    qualification_multiple_choice_per_epoch: int = field(
        default_factory=lambda: _env_int(
            "THINKER_QUALIFICATION_MULTIPLE_CHOICE_PER_EPOCH", 25
        )
    )
    multiple_choice_dataset: str = field(
        default_factory=lambda: _env_str(
            "THINKER_MULTIPLE_CHOICE_DATASET", "nvidia/OpenScienceReasoning-2"
        )
    )
    multiple_choice_split: str = field(
        default_factory=lambda: _env_str(
            "THINKER_MULTIPLE_CHOICE_SPLIT", "train"
        )
    )
    multiple_choice_max_new_tokens: int = field(
        default_factory=lambda: _env_int(
            "THINKER_MULTIPLE_CHOICE_MAX_NEW_TOKENS", 32768
        )
    )
    full_eval_top_k: int = field(default_factory=lambda: _env_int("THINKER_FULL_EVAL_TOP_K", 10))
    full_eval_history_weight: float = field(default_factory=lambda: _env_float("THINKER_FULL_EVAL_HISTORY_WEIGHT", 0.30))
    full_eval_ema_alpha: float = field(default_factory=lambda: _env_float("THINKER_FULL_EVAL_EMA_ALPHA", 0.80))
    champion_history_rounds: int = field(
        default_factory=lambda: _env_int("THINKER_CHAMPION_HISTORY_ROUNDS", 5)
    )
    full_eval_skip_after_rounds: int = field(
        default_factory=lambda: _env_int("THINKER_FULL_EVAL_SKIP_AFTER_ROUNDS", 5)
    )
    eval_cache_path: str = field(
        default_factory=lambda: _env_str("THINKER_EVAL_CACHE_PATH", ".thinker/validator/eval_cache.jsonl")
    )
    round_state_path: str = field(
        default_factory=lambda: _env_str("THINKER_ROUND_STATE_PATH", ".thinker/validator/round_state.json")
    )
    k_rollouts: int = field(default_factory=lambda: _env_int("THINKER_K_ROLLOUTS", 2))
    min_coverage_per_band: int = field(default_factory=lambda: _env_int("THINKER_MIN_COVERAGE_PER_BAND", 1))
    n_difficulty_bands: int = field(default_factory=lambda: _env_int("THINKER_N_DIFFICULTY_BANDS", 4))
    score_weight_math: float = field(
        default_factory=lambda: _env_float("THINKER_SCORE_WEIGHT_MATH", 0.50)
    )
    score_weight_long_context_qa: float = field(
        default_factory=lambda: _env_float("THINKER_SCORE_WEIGHT_LONG_CONTEXT_QA", 0.30)
    )
    score_weight_multiple_choice: float = field(
        default_factory=lambda: _env_float("THINKER_SCORE_WEIGHT_MULTIPLE_CHOICE", 0.20)
    )
    problem_weight_floor: float = field(
        default_factory=lambda: _env_float("THINKER_PROBLEM_WEIGHT_FLOOR", 0.05)
    )
    problem_weight_gamma: float = field(
        default_factory=lambda: _env_float("THINKER_PROBLEM_WEIGHT_GAMMA", 0.5)
    )

    synthesized_enabled: bool = field(
        default_factory=lambda: _env_bool("THINKER_SYNTHESIZED_ENABLED", True)
    )
    synthesized_dataset: str = field(
        default_factory=lambda: _env_str(
            "THINKER_SYNTHESIZED_DATASET", "nvidia/Nemotron-Math-v2"
        )
    )
    synthesized_split: str = field(
        default_factory=lambda: _env_str("THINKER_SYNTHESIZED_SPLIT", "high_part02")
    )
    synthesized_max_new_tokens: int = field(
        default_factory=lambda: _env_int("THINKER_SYNTHESIZED_MAX_NEW_TOKENS", 512)
    )
    synthesized_max_scan: int = field(
        default_factory=lambda: _env_int("THINKER_SYNTHESIZED_MAX_SCAN", 2_000)
    )

    epoch_blocks: int = field(default_factory=lambda: _env_int("THINKER_EPOCH_BLOCKS", 360))

    retrieval_cache_dir: str = field(
        default_factory=lambda: _env_str(
            "THINKER_RETRIEVAL_CACHE_DIR",
            str(Path.home() / ".thinker" / "retrieval"),
        )
    )
    retrieval_corpus_path: str = field(
        default_factory=lambda: _env_str("THINKER_RETRIEVAL_CORPUS_PATH", "")
    )
    retrieval_index_dir: str = field(
        default_factory=lambda: _env_str("THINKER_RETRIEVAL_INDEX_DIR", "")
    )
    retrieval_host: str = field(
        default_factory=lambda: _env_str("THINKER_RETRIEVAL_HOST", "127.0.0.1")
    )
    retrieval_port: int = field(default_factory=lambda: _env_int("THINKER_RETRIEVAL_PORT", 8765))
    retrieval_default_topk: int = field(
        default_factory=lambda: _env_int("THINKER_RETRIEVAL_DEFAULT_TOPK", 50)
    )
    retrieval_corpus_limit: int = field(
        default_factory=lambda: _env_int("THINKER_RETRIEVAL_CORPUS_LIMIT", 0)
    )
    retrieval_auto_download: bool = field(
        default_factory=lambda: _env_bool("THINKER_RETRIEVAL_AUTO_DOWNLOAD", True)
    )
    retrieval_mmap_index: bool = field(
        default_factory=lambda: _env_bool("THINKER_RETRIEVAL_MMAP_INDEX", True)
    )

    wandb_project: str = field(
        default_factory=lambda: _env_str(
            "THINKER_WANDB_PROJECT", "openvlcllm-fast-thinker/fast-thinker"
        )
    )
    wandb_entity: str = field(default_factory=lambda: _env_str("THINKER_WANDB_ENTITY", ""))

def load_config() -> ThinkerConfig:
    return ThinkerConfig()


def validate_adapter_bounds(
    config: ThinkerConfig, *, rank: Any, target_modules: Any, n_bytes: Any
) -> tuple[bool, str]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        return False, "rank must be a positive integer"
    if rank > config.max_lora_rank:
        return False, f"rank {rank} exceeds max_lora_rank {config.max_lora_rank}"
    if isinstance(n_bytes, bool) or not isinstance(n_bytes, int) or n_bytes < 0:
        return False, "adapter size must be a non-negative integer"
    if n_bytes > config.max_adapter_bytes:
        return False, f"adapter size {n_bytes} exceeds max_adapter_bytes {config.max_adapter_bytes}"
    if not isinstance(target_modules, list) or not target_modules:
        return False, "target_modules must be a non-empty list"
    if any(not isinstance(module, str) or not module for module in target_modules):
        return False, "target_modules must contain non-empty strings"
    disallowed = sorted(set(target_modules) - set(config.allowed_target_modules))
    if disallowed:
        return False, f"target_modules not allowed: {disallowed}"
    return True, ""
