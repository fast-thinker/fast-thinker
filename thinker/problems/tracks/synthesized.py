from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from thinker.problems.interface import (
    Difficulty,
    extract_final_boxed_answer,
    register_track,
)
from thinker.reward.verify import check_equivalence

DEFAULT_DATASET = "nvidia/Nemotron-Math-v2"
DEFAULT_SPLIT = "high_part02"
_PARQUET_REVISION = "refs/convert/parquet"
_PARQUET_CONFIG = "default"


def _resolve_parquet_paths(
    dataset_name: str,
    split: str,
    *,
    revision: str = _PARQUET_REVISION,
    config: str = _PARQUET_CONFIG,
    max_attempts: int = 4,
) -> list[str]:
    """List a split's parquet shard paths via one narrow, non-recursive call.

    Resolving a `*.parquet` glob through fsspec's HfFileSystem hits the HF
    tree-listing API on every fresh process, even when the shards are already
    in the local cache -- the wildcard has to be turned into concrete
    filenames before the local cache can be consulted at all. Listing just
    this one split directory (instead of letting fsspec do it) keeps that
    call cheap, and retrying with backoff absorbs the occasional transient
    5xx/timeout from that endpoint.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    path_in_repo = f"{config}/{split}"
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            entries = api.list_repo_tree(
                repo_id=dataset_name,
                repo_type="dataset",
                revision=revision,
                path_in_repo=path_in_repo,
                recursive=False,
            )
            paths = sorted(
                entry.path for entry in entries if entry.path.endswith(".parquet")
            )
            if not paths:
                raise ValueError(
                    f"no parquet files found under {dataset_name}@{revision}/{path_in_repo}"
                )
            return paths
        except Exception as exc:  # transient HF API/network errors
            last_exc = exc
            if attempt + 1 == max_attempts:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"failed to list parquet files for {dataset_name}@{revision}/{path_in_repo} "
        f"after {max_attempts} attempt(s): {last_exc}"
    ) from last_exc


def _cached_parquet_paths(dataset_name: str, split: str) -> list[str]:
    """Resolve a split's parquet shard paths, caching the result on local disk.

    Once resolved, the file list is reused across process restarts so we
    only need to ask Hugging Face "what files exist here" once per
    dataset/split, not once per validator startup.
    """
    from huggingface_hub import cached_assets_path

    cache_dir = cached_assets_path(
        library_name="thinker",
        namespace=dataset_name.replace("/", "--"),
        subfolder=split,
    )
    cache_file = cache_dir / "parquet_files.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached and all(isinstance(p, str) for p in cached):
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    paths = _resolve_parquet_paths(dataset_name, split)
    try:
        cache_file.write_text(json.dumps(paths), encoding="utf-8")
    except OSError:
        pass
    return paths


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(.+?)\s*\}", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*:?\s*", re.IGNORECASE)


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        ...


def _int_seed(seed: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _row_problem(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt", "input"):
        text = _clean_text(row.get(key))
        if text:
            return text
    raise ValueError("Nemotron-Math row has no problem field")


def _row_answer(row: dict[str, Any]) -> str:
    for key in ("expected_answer", "answer", "final_answer", "gold_answer"):
        text = _clean_text(row.get(key))
        if text:
            boxed = _BOXED_RE.search(text)
            if boxed:
                return boxed.group(1).strip()
            return _FINAL_ANSWER_RE.sub("", text, count=1).strip()
    raise ValueError("Nemotron-Math row has no expected answer field")


@dataclass(frozen=True)
class _SynthesizedSource:
    gold_answer: str
    source_problem: str
    dataset_index: int
    problem_id: str


@dataclass(frozen=True)
class _SynthesizedInstance(_SynthesizedSource):
    prompt: str
    transform: str


class SynthesizedTrack:
    track = "synthesized"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        dataset_name: str = DEFAULT_DATASET,
        split: str = DEFAULT_SPLIT,
        max_scan: int = 2_000,
    ):
        self._dataset_name = dataset_name
        self._split = split
        self._max_scan = max(1, int(max_scan))
        self._dataset = None
        self._source_cache: dict[str, _SynthesizedSource] = {}
        self._cache: dict[str, _SynthesizedInstance] = {}

    def _data(self):
        if self._dataset is None:
            from datasets import load_dataset

            # Loading the dataset repository by name prepares every split before
            # returning the requested one. Point the generic Parquet loader at the
            # converted files for this split instead, so sibling splits are never
            # downloaded. Resolving the exact shard filenames ourselves (and
            # caching that resolution -- see _cached_parquet_paths) means we
            # don't need fsspec's `*.parquet` glob, which re-hits the HF
            # tree-listing API on every fresh process even when the shards are
            # already cached locally.
            if self._dataset_name == DEFAULT_DATASET:
                relative_paths = _cached_parquet_paths(self._dataset_name, self._split)
                parquet_files = [
                    f"hf://datasets/{self._dataset_name}@{_PARQUET_REVISION}/{path}"
                    for path in relative_paths
                ]
                self._dataset = load_dataset(
                    "parquet",
                    data_files={self._split: parquet_files},
                    split=self._split,
                )
            else:
                # Preserve support for custom datasets whose config name or file
                # layout may not match Nemotron-Math-v2.
                self._dataset = load_dataset(self._dataset_name, split=self._split)
        return self._dataset

    def _select_row(self, seed: str) -> tuple[int, dict[str, Any]]:
        data = self._data()
        dataset_size = len(data)
        if dataset_size <= 0:
            raise ValueError(f"{self._dataset_name}@{self._split} is empty")
        rng = random.Random(_int_seed(seed, "nemotron-row"))
        tried: set[int] = set()
        while len(tried) < min(dataset_size, self._max_scan):
            index = rng.randrange(dataset_size)
            if index in tried:
                continue
            tried.add(index)
            row = dict(data[index])
            try:
                _row_problem(row)
                _row_answer(row)
            except ValueError:
                continue
            return index, row
        raise ValueError(
            f"{self._dataset_name}@{self._split} yielded no usable row after "
            f"checking {len(tried)} deterministic candidates"
        )

    def _source(self, seed: str) -> _SynthesizedSource:
        cached = self._source_cache.get(seed)
        if cached is not None:
            return cached
        index, row = self._select_row(seed)
        source = _SynthesizedSource(
            gold_answer=_row_answer(row),
            source_problem=_row_problem(row),
            dataset_index=index,
            problem_id=_clean_text(row.get("problem_id") or row.get("id") or index),
        )
        self._source_cache[seed] = source
        return source

    def _generate(self, seed: str) -> _SynthesizedInstance:
        source = self._source(seed)
        return _SynthesizedInstance(
            gold_answer=source.gold_answer,
            source_problem=source.source_problem,
            dataset_index=source.dataset_index,
            problem_id=source.problem_id,
            prompt=source.source_problem,
            transform="dataset_problem",
        )

    def _instance(self, seed: str) -> _SynthesizedInstance:
        if seed not in self._cache:
            self._cache[seed] = self._generate(seed)
        return self._cache[seed]

    def render(self, seed: str) -> str:
        return self._instance(seed).prompt

    def verify(self, seed: str, output: str) -> bool:
        gold = self._instance(seed).gold_answer
        boxed = extract_final_boxed_answer(output)
        if boxed is None:
            return False
        try:
            return check_equivalence(rf"\boxed{{{gold}}}", rf"\boxed{{{boxed}}}")
        except ValueError:
            return False

    def difficulty(self, seed: str) -> Difficulty:
        inst = self._instance(seed)
        return Difficulty(
            track=self.track,
            params={
                "source": "nemotron_math_v2",
                "dataset": self._dataset_name,
                "split": self._split,
                "dataset_index": inst.dataset_index,
                "problem_id": inst.problem_id,
                "transform": inst.transform,
            },
        )

    def min_tokens(self, seed: str) -> int:
        inst = self._instance(seed)
        return max(128, len(inst.source_problem) // 4 + 64)


def register(llm_client: LLMClient | None = None, **kwargs) -> SynthesizedTrack:
    track = SynthesizedTrack(llm_client, **kwargs)
    register_track(track)
    return track
