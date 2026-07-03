from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from thinker.reward.relative import (
    peer_completion_efficiency_rewards,
    relative_reasoning_reward,
)
from thinker.validator.scoring import RolloutResult

DATASET_NAME = "nvidia/OpenScienceReasoning-2"
DEFAULT_SPLIT = "train"
MULTIPLE_CHOICE_BAND = "multiple_choice"
CHOICE_RE = re.compile(r"\b([A-Z])\s*[:.)]")
CHOICE_ANSWER_RE = re.compile(r"^[A-Z](?:\s*[,/]\s*[A-Z])*$")
OPTION_MARKER_RE = re.compile(
    r"(?<!\S)(?P<label>[A-Z])(?P<delimiter>[.:)])(?P<space>[ \t]+)"
)
MULTIPLE_CHOICE_SYSTEM_PROMPT = (
    "Solve the multiple-choice problem carefully. After reasoning, end your response "
    "with exactly one of the displayed option letters in LaTeX boxed form. Do not "
    "put words or multiple letters inside the box."
)


@dataclass(frozen=True)
class MultipleChoiceInstance:
    seed: str
    prompt: str
    ground_truth: str
    problem_id: str
    enable_thinking: bool = True


@dataclass(frozen=True)
class MultipleChoiceAnswer:
    text: str
    completion_len: int
    verified: bool
    response: str = ""


@dataclass(frozen=True)
class MultipleChoiceMinerResult:
    score: float
    original: MultipleChoiceAnswer
    miner: MultipleChoiceAnswer


def _int_seed(seed: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _ground_truth(row: dict[str, Any]) -> str:
    if row.get("expected_answer") is not None:
        return str(row["expected_answer"])
    verification = _parse_mapping(row.get("verification_info"))
    if "ground_truth" in verification:
        return str(verification["ground_truth"])
    solution = _parse_mapping(row.get("gold_standard_solution"))
    if "output" in solution:
        return str(solution["output"])
    raise ValueError("multiple-choice row has no ground_truth/output field")


def _extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(text):
        ch = text[index]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        index += 1
    return None


def extract_boxed_answer(text: str) -> str:
    boxed = _extract_last_boxed(text)
    return boxed if boxed is not None else text.strip()


def _normalize_answer(text: str) -> str:
    text = extract_boxed_answer(text)
    leading_choice = re.match(r"^\s*([A-Z])\s*[:.)]", text.strip(), re.IGNORECASE)
    if leading_choice:
        return leading_choice.group(1).upper()
    normalized = re.sub(r"\s+", "", text.strip())
    if CHOICE_ANSWER_RE.fullmatch(normalized.upper()):
        return normalized.upper()
    return text.strip()


def _row_input(row: dict[str, Any]) -> str:
    for key in ("input", "prompt", "question"):
        if row.get(key) is not None:
            text = str(row[key]).strip()
            if text:
                return text
    raise ValueError("qualification row has no input/prompt/question field")


def _answer_labels(answer: str) -> list[str]:
    normalized = _normalize_answer(answer)
    if CHOICE_ANSWER_RE.fullmatch(normalized) is None:
        return []
    return re.findall(r"[A-Z]", normalized)


def _option_block(
    prompt: str, ground_truth: str
) -> tuple[str, list[tuple[str, str]]] | None:
    matches = list(OPTION_MARKER_RE.finditer(prompt))
    answer_labels = set(_answer_labels(ground_truth))
    if len(matches) < 2 or not answer_labels:
        return None

    # Science rows use one final A/B/C/... option block. Requiring consecutive
    # source labels avoids treating incidental lettered lines as answer options.
    runs: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in matches:
        label = match.group("label")
        expected = chr(ord("A") + len(current)) if current else "A"
        if label != expected:
            if len(current) >= 2:
                runs.append(current)
            current = [match] if label == "A" else []
        else:
            current.append(match)
    if len(current) >= 2:
        runs.append(current)

    candidates = [
        run for run in runs if answer_labels.issubset({item.group("label") for item in run})
    ]
    if not candidates:
        return None
    run = candidates[-1]
    options: list[tuple[str, str]] = []
    for index, match in enumerate(run):
        end = run[index + 1].start() if index + 1 < len(run) else len(prompt)
        text = prompt[match.end() : end].strip()
        if not text:
            return None
        options.append((match.group("label"), text))
    return prompt[: run[0].start()].rstrip(), options


def _shuffle_options(prompt: str, ground_truth: str, seed: str) -> tuple[str, str]:
    parsed = _option_block(prompt, ground_truth)
    if parsed is None:
        raise ValueError("multiple-choice row has no supported A/B/C/... option block")
    stem, options = parsed
    if len(options) > len(string.ascii_uppercase):
        raise ValueError("multiple-choice row has too many options")

    rng = random.Random(_int_seed(seed, "multiple_choice_option_layout"))
    shuffled = list(options)
    rng.shuffle(shuffled)
    original_labels = [label for label, _text in options]
    unused_labels = [
        label for label in string.ascii_uppercase if label not in original_labels
    ]
    label_pool = (
        unused_labels
        if len(unused_labels) >= len(shuffled)
        else list(string.ascii_uppercase)
    )
    labels = rng.sample(label_pool, len(shuffled))

    remapped = {
        original_label: display_label
        for display_label, (original_label, _text) in zip(labels, shuffled)
    }
    answer_labels = _answer_labels(ground_truth)
    separator = "," if "," in _normalize_answer(ground_truth) else "/"
    shuffled_ground_truth = separator.join(remapped[label] for label in answer_labels)
    rendered_options = "\n".join(
        f"{display_label}) {text}"
        for display_label, (_original_label, text) in zip(labels, shuffled)
    )
    return f"{stem}\n\n{rendered_options}", shuffled_ground_truth


def _is_multiple_choice_row(row: dict[str, Any]) -> bool:
    try:
        answer = _normalize_answer(_ground_truth(row))
        prompt = _row_input(row)
    except Exception:
        return False
    return bool(
        CHOICE_ANSWER_RE.fullmatch(answer)
        and CHOICE_RE.search(prompt)
        and _option_block(prompt, answer) is not None
    )


class MultipleChoiceEvaluator:
    def __init__(
        self,
        inference,
        *,
        dataset_name: str = DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        rows: Sequence[dict[str, Any]] | None = None,
        max_new_tokens: int | None = 32768,
    ):
        self._inference = inference
        self._dataset_name = dataset_name
        self._split = split
        self._rows = list(rows) if rows is not None else None
        self._dataset = None
        self._max_new_tokens = max_new_tokens
        self._used_problem_ids: set[str] = set()
        self._previous_problem_ids: set[str] = set()
        self._used_dataset_indices: set[int] = set()

    def _load_dataset(self):
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except Exception as exc:
                raise RuntimeError(
                    "multiple-choice qualification requires the `datasets` package"
                ) from exc
            self._dataset = load_dataset(
                self._dataset_name,
                split=self._split,
            )
            if not callable(getattr(self._dataset, "select", None)):
                raise TypeError(
                    "multiple-choice qualification requires a map-style dataset with select()"
                )
        return self._dataset

    def _select_provided_rows(
        self, seeds: list[str], *, common_prefix_count: int = 0
    ) -> list[tuple[int, dict[str, Any], str]]:
        rows = [row for row in (self._rows or []) if _is_multiple_choice_row(row)]
        if not rows:
            raise ValueError("qualification multiple-choice dataset is empty")
        if len(rows) < len(seeds):
            raise ValueError(
                f"qualification requires {len(seeds)} unique rows, found {len(rows)}"
            )

        indexed_rows = [
            (index, row, str(row.get("problem_id", row.get("id", index))))
            for index, row in enumerate(rows)
        ]
        common_count = min(len(seeds), max(0, int(common_prefix_count)))
        selected: list[tuple[int, dict[str, Any], str]] = []
        selected_problem_ids: set[str] = set()
        common_available = list(indexed_rows)
        for seed in seeds[:common_count]:
            rng = random.Random(_int_seed(seed, "multiple_choice"))
            selected_index = rng.randrange(len(common_available))
            item = common_available.pop(selected_index)
            selected.append(item)
            selected_problem_ids.add(item[2])

        available = [
            item for item in indexed_rows if item[2] not in self._used_problem_ids
            and item[2] not in selected_problem_ids
        ]
        private_count = len(seeds) - common_count
        if len(available) < private_count:
            self._used_problem_ids.clear()
            available = [
                item
                for item in indexed_rows
                if item[2] not in self._previous_problem_ids
                and item[2] not in selected_problem_ids
            ]
        if len(available) < private_count:
            available = [
                item for item in indexed_rows if item[2] not in selected_problem_ids
            ]

        for seed in seeds[common_count:]:
            rng = random.Random(_int_seed(seed, "multiple_choice"))
            selected_index = rng.randrange(len(available))
            index, row, problem_id = available.pop(selected_index)
            selected_problem_ids.add(problem_id)
            selected.append((index, row, problem_id))
        self._used_problem_ids.update(selected_problem_ids)
        self._previous_problem_ids = selected_problem_ids
        return selected

    def _select_dataset_rows(
        self, seeds: list[str], *, common_prefix_count: int = 0
    ) -> list[tuple[int, dict[str, Any], str]]:
        common_count = min(len(seeds), max(0, int(common_prefix_count)))
        if common_count:
            previous_used = set(self._used_dataset_indices)
            self._used_dataset_indices.clear()
            try:
                common = self._select_dataset_rows(
                    seeds[:common_count], common_prefix_count=0
                )
            except Exception:
                self._used_dataset_indices = previous_used
                raise
            common_indices = {index for index, _row, _problem_id in common}
            self._used_dataset_indices = previous_used | common_indices
            private = self._select_dataset_rows(
                seeds[common_count:], common_prefix_count=0
            )
            self._used_dataset_indices.update(common_indices)
            return common + private

        dataset = self._load_dataset()
        dataset_size = len(dataset)
        target = len(seeds)
        if dataset_size < target:
            raise ValueError(
                f"qualification requires {target} unique rows, dataset has {dataset_size}"
            )
        if dataset_size - len(self._used_dataset_indices) < target:
            self._used_dataset_indices.clear()

        rng = random.Random(_int_seed("|".join(seeds), "dataset_indices"))
        tried: set[int] = set()
        selected: list[tuple[int, dict[str, Any], str]] = []
        max_scanned = min(dataset_size, max(1_000, target * 100))
        while len(selected) < target and len(tried) < max_scanned:
            batch_size = min(max(32, (target - len(selected)) * 4), max_scanned - len(tried))
            candidate_indices: list[int] = []
            while len(candidate_indices) < batch_size and len(tried) < max_scanned:
                index = rng.randrange(dataset_size)
                if index in tried or index in self._used_dataset_indices:
                    continue
                tried.add(index)
                candidate_indices.append(index)
            if not candidate_indices:
                break
            candidate_rows = dataset.select(candidate_indices)
            for index, raw_row in zip(candidate_indices, candidate_rows):
                row = dict(raw_row)
                if not _is_multiple_choice_row(row):
                    continue
                problem_id = str(row.get("problem_id", row.get("id", index)))
                selected.append((index, row, problem_id))
                if len(selected) >= target:
                    break

        if len(selected) < target:
            raise ValueError(
                "multiple-choice qualification found only "
                f"{len(selected)} eligible row(s) after randomly selecting "
                f"{len(tried)} indices; required {target}. Check the dataset schema."
            )
        self._used_dataset_indices.update(index for index, _row, _problem_id in selected)
        return selected

    def generate_instances(
        self,
        seeds: Iterable[str],
        *,
        thinking_samples: int | None = None,
        common_prefix_count: int = 0,
    ) -> list[MultipleChoiceInstance]:
        seeds = list(seeds)
        if not seeds:
            return []
        thinking_count = (
            len(seeds)
            if thinking_samples is None
            else min(len(seeds), max(0, int(thinking_samples)))
        )
        selected = (
            self._select_provided_rows(
                seeds, common_prefix_count=common_prefix_count
            )
            if self._rows is not None
            else self._select_dataset_rows(
                seeds, common_prefix_count=common_prefix_count
            )
        )
        instances: list[MultipleChoiceInstance] = []
        for index, (seed, (_row_index, row, problem_id)) in enumerate(
            zip(seeds, selected)
        ):
            prompt, ground_truth = _shuffle_options(
                _row_input(row), _ground_truth(row), seed
            )
            instances.append(
                MultipleChoiceInstance(
                    seed=seed,
                    prompt=prompt,
                    ground_truth=ground_truth,
                    problem_id=problem_id,
                    enable_thinking=index < thinking_count,
                )
            )
        return instances

    def _generate_original(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        enable_thinking: bool = True,
    ) -> list[tuple[str, int]]:
        return self._inference.generate_original_limited(
            prompts,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            system_prompt=MULTIPLE_CHOICE_SYSTEM_PROMPT,
        )

    def _generate_miner(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        enable_thinking: bool = True,
    ) -> list[tuple[str, int]]:
        return self._inference.generate_limited(
            miner_id,
            adapter_files,
            prompts,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            system_prompt=MULTIPLE_CHOICE_SYSTEM_PROMPT,
        )

    @staticmethod
    def _answer(completion: str, token_count: int, ground_truth: str) -> MultipleChoiceAnswer:
        text = extract_boxed_answer(completion)
        return MultipleChoiceAnswer(
            text=text,
            completion_len=token_count,
            verified=_normalize_answer(text) == _normalize_answer(ground_truth),
            response=completion,
        )

    def score_original_batch(
        self, instances: list[MultipleChoiceInstance]
    ) -> list[MultipleChoiceAnswer]:
        if not instances:
            return []
        completions: list[tuple[str, int] | None] = [None] * len(instances)
        for enable_thinking in (True, False):
            indexes = [
                index
                for index, instance in enumerate(instances)
                if instance.enable_thinking is enable_thinking
            ]
            if not indexes:
                continue
            generated = self._generate_original(
                [instances[index].prompt for index in indexes],
                max_new_tokens=self._max_new_tokens,
                enable_thinking=enable_thinking,
            )
            if len(generated) != len(indexes):
                raise ValueError(
                    "original model must return one multiple-choice answer per prompt"
                )
            for index, completion in zip(indexes, generated):
                completions[index] = completion
        if any(completion is None for completion in completions):
            raise ValueError("original model must return one multiple-choice answer per prompt")
        answers: list[MultipleChoiceAnswer] = []
        for instance, completion in zip(instances, completions):
            if completion is None:
                raise ValueError(
                    "original model must return one multiple-choice answer per prompt"
                )
            answers.append(
                self._answer(completion[0], completion[1], instance.ground_truth)
            )
        return answers

    def score_miner_batch(
        self,
        *,
        miner_id: str,
        adapter_files: dict[str, bytes],
        instances: list[MultipleChoiceInstance],
        originals: list[MultipleChoiceAnswer],
    ) -> list[MultipleChoiceMinerResult]:
        if len(instances) != len(originals):
            raise ValueError("instances and originals must match")
        if not instances:
            return []
        budgets = [max(1, original.completion_len) for original in originals]
        completions: list[tuple[str, int] | None] = [None] * len(instances)
        grouped: dict[tuple[int, bool], list[int]] = {}
        for index, (budget, instance) in enumerate(zip(budgets, instances)):
            grouped.setdefault((budget, instance.enable_thinking), []).append(index)
        for (budget, enable_thinking), indexes in grouped.items():
            chunk = [instances[index].prompt for index in indexes]
            chunk_completions = self._generate_miner(
                miner_id,
                adapter_files,
                chunk,
                max_new_tokens=budget,
                enable_thinking=enable_thinking,
            )
            if len(chunk_completions) != len(indexes):
                raise ValueError("miner must return one multiple-choice answer per prompt")
            for index, completion in zip(indexes, chunk_completions):
                completions[index] = completion
        results: list[MultipleChoiceMinerResult] = []
        for instance, original, completion in zip(instances, originals, completions):
            if completion is None:
                raise ValueError("missing multiple-choice miner completion")
            miner = self._answer(completion[0], completion[1], instance.ground_truth)
            results.append(
                MultipleChoiceMinerResult(
                    score=relative_reasoning_reward(
                        original_verified=original.verified,
                        miner_verified=miner.verified,
                        original_completion_len=original.completion_len,
                        miner_completion_len=miner.completion_len,
                    ),
                    original=original,
                    miner=miner,
                )
            )
        return results

    def score_miners_batch(
        self,
        *,
        miners: list[tuple[str, dict[str, bytes]]],
        instances: list[MultipleChoiceInstance],
        originals: list[MultipleChoiceAnswer],
    ) -> dict[str, list[MultipleChoiceMinerResult]]:
        if len(instances) != len(originals):
            raise ValueError("instances and originals must match")
        if not miners:
            return {}
        if not instances:
            return {miner_id: [] for miner_id, _adapter_files in miners}

        budgets = [max(1, original.completion_len) for original in originals]
        completions_by_miner: dict[str, list[tuple[str, int] | None]] = {
            miner_id: [None] * len(instances) for miner_id, _adapter_files in miners
        }
        for enable_thinking in (True, False):
            indexes = [
                index
                for index, instance in enumerate(instances)
                if instance.enable_thinking is enable_thinking
            ]
            if not indexes:
                continue
            requests = [
                (miner_id, adapter_files, instances[index].prompt)
                for miner_id, adapter_files in miners
                for index in indexes
            ]
            max_new_tokens_list = [
                budgets[index]
                for _miner_id, _adapter_files in miners
                for index in indexes
            ]
            flat_completions = self._inference.generate_for_miners_batch(
                requests,
                max_new_tokens_list=max_new_tokens_list,
                enable_thinking=enable_thinking,
                system_prompt=MULTIPLE_CHOICE_SYSTEM_PROMPT,
            )
            if len(flat_completions) != len(requests):
                raise ValueError(
                    "batched multiple-choice generation must return one answer per request"
                )
            cursor = 0
            for miner_id, _adapter_files in miners:
                for index in indexes:
                    completions_by_miner[miner_id][index] = flat_completions[cursor]
                    cursor += 1

        results_by_miner: dict[str, list[MultipleChoiceMinerResult]] = {}
        answers_by_miner: dict[str, list[MultipleChoiceAnswer]] = {}
        for miner_id, _adapter_files in miners:
            answers: list[MultipleChoiceAnswer] = []
            for instance, original, completion in zip(
                instances, originals, completions_by_miner[miner_id]
            ):
                if completion is None:
                    raise ValueError("missing batched multiple-choice miner completion")
                text, token_count = completion
                answers.append(self._answer(text, token_count, instance.ground_truth))
            answers_by_miner[miner_id] = answers

        rewards_by_miner: dict[str, list[float]] = {
            miner_id: [0.0] * len(instances) for miner_id, _adapter_files in miners
        }
        miner_ids = [miner_id for miner_id, _adapter_files in miners]
        for index, original in enumerate(originals):
            base_rewards = [
                relative_reasoning_reward(
                    original_verified=original.verified,
                    miner_verified=answers_by_miner[miner_id][index].verified,
                    original_completion_len=original.completion_len,
                    miner_completion_len=answers_by_miner[miner_id][index].completion_len,
                )
                for miner_id in miner_ids
            ]
            rewards = peer_completion_efficiency_rewards(
                original_verified=original.verified,
                miner_verified=[
                    answers_by_miner[miner_id][index].verified
                    for miner_id in miner_ids
                ],
                miner_completion_lens=[
                    answers_by_miner[miner_id][index].completion_len
                    for miner_id in miner_ids
                ],
                base_rewards=base_rewards,
            )
            for miner_id, reward in zip(miner_ids, rewards):
                rewards_by_miner[miner_id][index] = reward

        for miner_id, _adapter_files in miners:
            results_by_miner[miner_id] = [
                MultipleChoiceMinerResult(
                    score=score,
                    original=original,
                    miner=miner,
                )
                for original, miner, score in zip(
                    originals,
                    answers_by_miner[miner_id],
                    rewards_by_miner[miner_id],
                )
            ]
        return results_by_miner


def rollout_result(
    instance: MultipleChoiceInstance,
    result: MultipleChoiceMinerResult,
    *,
    sample_weight: float = 1.0,
) -> RolloutResult:
    return RolloutResult(
        track=MULTIPLE_CHOICE_BAND,
        seed=instance.seed,
        band=MULTIPLE_CHOICE_BAND,
        score=result.score,
        sample_weight=sample_weight,
        original_verified=result.original.verified,
        miner_verified=result.miner.verified,
        original_completion_len=result.original.completion_len,
        miner_completion_len=result.miner.completion_len,
    )
