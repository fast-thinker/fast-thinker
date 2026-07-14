from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from tqdm.auto import tqdm

from thinker.config import ThinkerConfig
from thinker.common_seed import build_sample_seed_plan
from thinker.problems.decontam import DecontaminationStore
from thinker.problems.interface import (
    ProblemInstance,
    get_track,
    registered_tracks,
    render_instance,
)
from thinker.reward.relative import (
    peer_completion_efficiency_rewards,
    relative_reasoning_reward,
)
from thinker.submission.adapter_validation import ValidatedAdapter, validate_adapter_files
from thinker.submission.crypto import (
    EncryptedSubmission,
    content_hash,
    decrypt_as_recipient,
    max_encrypted_adapter_ciphertext_bytes,
    unpack_adapter_bundle,
)
from thinker.submission.fingerprint import LoraFingerprint, compute_lora_fingerprint, fingerprints_collide
from thinker.validator.long_context_qa import (
    LongContextAnswer,
    LongContextMinerResult,
    LongContextQAEvaluator,
    LongContextQAInstance,
)
from thinker.validator.eval_cache import EvaluationCache
from thinker.validator.round_state import RoundStateStore
from thinker.validator.scoring import RolloutResult, StratifiedScore, difficulty_band, stratified_score
from thinker.validator.multiple_choice import (
    MultipleChoiceAnswer,
    MultipleChoiceEvaluator,
    MultipleChoiceInstance,
    MultipleChoiceMinerResult,
    rollout_result as multiple_choice_rollout_result,
)

TASK_MATH = "math"
TASK_LONG_CONTEXT_QA = "long_context_qa"
TASK_MULTIPLE_CHOICE = "multiple_choice"
MATH_SYSTEM_PROMPT = (
    "Solve the math problem carefully. End with exactly one final answer in LaTeX "
    "\\boxed{...} form and write nothing after it. Do not use \\boxed anywhere else."
)


@dataclass(frozen=True)
class MinerSubmissionPointer:
    miner_id: str
    epoch: int
    repo_id: str
    sha256: str


class SubmissionTransport(Protocol):
    def fetch(self, pointer: MinerSubmissionPointer) -> EncryptedSubmission: ...


class InferenceBackend(Protocol):
    def generate_original(self, prompts: list[str]) -> list[tuple[str, int]]:
        ...

    def generate_original_limited(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
        stop: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        ...

    def generate_limited(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        ...

    def generate_for_miners_batch(
        self,
        requests: list[tuple[str, dict[str, bytes], str]],
        *,
        max_new_tokens_list: list[int | None] | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        ...

    def suppress_progress(self):
        ...


class WeightSetter(Protocol):
    def set_weights(self, scores: dict[str, float]) -> None: ...


@dataclass
class MinerEpochResult:
    miner_id: str
    score: StratifiedScore | None
    rejected_reason: str | None = None
    completion_len: float | None = None
    correctness_score: float | None = None
    original_score: float | None = None
    original_correctness_score: float | None = None
    original_completion_len: float | None = None
    telemetry_count: int = 0
    task_scores: dict[str, float] = field(default_factory=dict)
    task_completion_len: dict[str, float] = field(default_factory=dict)
    task_correctness_score: dict[str, float] = field(default_factory=dict)
    task_original_score: dict[str, float] = field(default_factory=dict)
    task_original_correctness_score: dict[str, float] = field(default_factory=dict)
    task_original_completion_len: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OriginalRollout:
    verified: bool
    completion_len: int


@dataclass(frozen=True)
class PreparedMinerSubmission:
    miner_id: str
    pointer: MinerSubmissionPointer
    adapter_files: dict[str, bytes]
    adapter_hash: str


@dataclass(frozen=True)
class EpochBatch:
    math: list[ProblemInstance]
    long_context_qa: list[LongContextQAInstance]


@dataclass
class QualificationScoringCache:
    math_completions: dict[str, list[tuple[str, int]]]
    math_scores: dict[str, list[float]]
    math_errors: dict[str, str]
    long_context_results: dict[str, list[LongContextMinerResult]]
    long_context_errors: dict[str, str]
    multiple_choice_results: dict[str, list[MultipleChoiceMinerResult]]
    multiple_choice_errors: dict[str, str]
    sample_weights: dict[tuple[str, str], float]


class EpochLoop:
    def __init__(
        self,
        config: ThinkerConfig,
        recipient_id: str,
        recipient_privkey: bytes,
        transport: SubmissionTransport,
        inference: InferenceBackend,
        weight_setter: WeightSetter,
        decontam_store: DecontaminationStore,
        seed_fn,
        long_context_evaluator: LongContextQAEvaluator | None = None,
        multiple_choice_evaluator: MultipleChoiceEvaluator | None = None,
        known_fingerprints: dict[str, LoraFingerprint] | None = None,
        evaluation_cache: EvaluationCache | None = None,
        round_state: RoundStateStore | None = None,
        show_progress: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        fingerprint_exempt_miner_ids: set[str] | None = None,
    ):
        self._config = config
        self._recipient_id = recipient_id
        self._recipient_privkey = recipient_privkey
        self._transport = transport
        self._inference = inference
        self._weights = weight_setter
        self._decontam = decontam_store
        self._seed_fn = seed_fn
        self._long_context = long_context_evaluator
        self._multiple_choice = multiple_choice_evaluator
        self._fingerprints: dict[str, LoraFingerprint] = dict(known_fingerprints or {})
        self._fingerprint_exempt_miner_ids = frozenset(
            fingerprint_exempt_miner_ids or ()
        )
        self._eval_cache = evaluation_cache if evaluation_cache is not None else EvaluationCache(config.eval_cache_path)
        self._round_state = (
            round_state
            if round_state is not None
            else RoundStateStore(
                config.round_state_path,
                champion_history_rounds=config.champion_history_rounds,
            )
        )
        self._batched_math_failure_logged = False
        self._batched_long_context_failure_logged = False
        self._batched_multiple_choice_failure_logged = False
        self._show_progress = show_progress
        self._progress_callback = progress_callback

    @contextmanager
    def _progress(
        self,
        total: int,
        description: str,
        *,
        unit: str = "item",
        suppress_inference: bool = False,
    ):
        progress_context = (
            self._inference.suppress_progress()
            if suppress_inference
            else nullcontext()
        )
        with progress_context, tqdm(
            total=total,
            desc=f"[thinker-validator] {description}",
            unit=unit,
            dynamic_ncols=True,
            disable=not self._show_progress,
        ) as progress:
            yield progress

    @staticmethod
    def _log(message: str) -> None:
        print(f"[thinker-validator] {message}", flush=True)

    def _report_progress(
        self,
        epoch: int,
        stage: str,
        stage_status: str,
        miner_updates: dict[str, str] | None = None,
        baseline_status: str | None = None,
        miner_scores: dict[str, float] | None = None,
    ) -> None:
        if self._progress_callback is None:
            return
        event = {
            "epoch": epoch,
            "stage": stage,
            "stage_status": stage_status,
            "miner_updates": dict(miner_updates or {}),
        }
        if baseline_status is not None:
            event["baseline_status"] = baseline_status
        if miner_scores is not None:
            event["miner_scores"] = dict(miner_scores)
        try:
            self._progress_callback(event)
        except Exception as exc:
            self._log(f"progress telemetry failed: {type(exc).__name__}")

    @staticmethod
    def _score_summary(result: MinerEpochResult) -> str:
        if result.score is None:
            return f"rejected: {result.rejected_reason}"
        coverage = "coverage-ok" if result.score.coverage_ok else "coverage-missing"
        return f"score={result.score.overall:.4f} ({coverage}; {result.score.reason or 'ok'})"

    def _score_type_weights(self) -> dict[str, float]:
        return {
            TASK_MATH: max(0.0, float(self._config.score_weight_math)),
            TASK_LONG_CONTEXT_QA: max(
                0.0, float(self._config.score_weight_long_context_qa)
            ),
            TASK_MULTIPLE_CHOICE: max(
                0.0, float(self._config.score_weight_multiple_choice)
            ),
        }

    @staticmethod
    def _rollout_task(result: RolloutResult) -> str:
        if result.track == TASK_LONG_CONTEXT_QA or result.band == TASK_LONG_CONTEXT_QA:
            return TASK_LONG_CONTEXT_QA
        if result.track == TASK_MULTIPLE_CHOICE or result.band == TASK_MULTIPLE_CHOICE:
            return TASK_MULTIPLE_CHOICE
        return TASK_MATH

    @staticmethod
    def _weighted_task_mean(
        values: dict[str, float],
        weights: dict[str, float],
    ) -> float | None:
        available = {
            task: float(value)
            for task, value in values.items()
            if value is not None
        }
        if not available:
            return None
        total_weight = sum(max(0.0, weights.get(task, 0.0)) for task in available)
        if total_weight <= 0:
            return sum(available.values()) / len(available)
        return sum(
            value * max(0.0, weights.get(task, 0.0))
            for task, value in available.items()
        ) / total_weight

    def _problem_weight(self, correct_count: int, total_count: int) -> float:
        floor = min(1.0, max(0.0, float(self._config.problem_weight_floor)))
        gamma = max(0.0, float(self._config.problem_weight_gamma))
        total = max(0, int(total_count))
        correct = min(total, max(0, int(correct_count)))
        if total <= 0 or correct == 0 or correct == total:
            return floor
        p_correct = correct / total
        return floor + (1.0 - floor) * ((1.0 - p_correct) ** gamma)

    def _math_sample_weights(
        self,
        batch: list[ProblemInstance],
        completions_by_miner: dict[str, list[tuple[str, int]]],
    ) -> dict[tuple[str, str], float]:
        weights: dict[tuple[str, str], float] = {}
        for index, instance in enumerate(batch):
            total = 0
            correct = 0
            track = get_track(instance.track)
            for completions in completions_by_miner.values():
                if index >= len(completions):
                    continue
                total += 1
                if track.verify(instance.seed, completions[index][0]):
                    correct += 1
            weights[(instance.track, instance.seed)] = self._problem_weight(
                correct, total
            )
        return weights

    def _long_context_sample_weights(
        self,
        batch: list[LongContextQAInstance],
        results_by_miner: dict[str, list[LongContextMinerResult]],
    ) -> dict[tuple[str, str], float]:
        weights: dict[tuple[str, str], float] = {}
        for index, instance in enumerate(batch):
            total = 0
            correct = 0
            for results in results_by_miner.values():
                if index >= len(results):
                    continue
                total += 1
                if results[index].miner.verified:
                    correct += 1
            weights[(TASK_LONG_CONTEXT_QA, instance.seed)] = self._problem_weight(
                correct, total
            )
        return weights

    def _multiple_choice_sample_weights(
        self,
        batch: list[MultipleChoiceInstance],
        results_by_miner: dict[str, list[MultipleChoiceMinerResult]],
    ) -> dict[tuple[str, str], float]:
        weights: dict[tuple[str, str], float] = {}
        for index, instance in enumerate(batch):
            total = 0
            correct = 0
            for results in results_by_miner.values():
                if index >= len(results):
                    continue
                total += 1
                if results[index].miner.verified:
                    correct += 1
            weights[(TASK_MULTIPLE_CHOICE, instance.seed)] = self._problem_weight(
                correct, total
            )
        return weights

    @staticmethod
    def _sample_weight(
        sample_weights: dict[tuple[str, str], float] | None,
        track: str,
        seed: str,
    ) -> float:
        if sample_weights is None:
            return 1.0
        return float(sample_weights.get((track, seed), 1.0))

    def _score_rollouts_by_task(
        self,
        results: list[RolloutResult],
        *,
        min_coverage_per_band: int,
    ) -> tuple[StratifiedScore, dict[str, float]]:
        by_task: dict[str, list[RolloutResult]] = defaultdict(list)
        for result in results:
            by_task[self._rollout_task(result)].append(result)
        if not by_task:
            return (
                StratifiedScore(overall=0.0, per_band={}, coverage_ok=True, reason=None),
                {},
            )

        task_scores: dict[str, float] = {}
        per_band: dict[str, float] = {}
        coverage_ok = True
        reasons: list[str] = []
        for task, task_results in sorted(by_task.items()):
            if task == TASK_MATH:
                score = stratified_score(
                    task_results,
                    min_coverage_per_band=min_coverage_per_band,
                )
            else:
                score = stratified_score(
                    task_results,
                    min_coverage_per_band=1,
                    required_bands={task},
                )
            task_scores[task] = score.overall
            per_band[f"type/{task}"] = score.overall
            per_band.update(
                {
                    f"{task}/{band}": value
                    for band, value in sorted(score.per_band.items())
                }
            )
            coverage_ok = coverage_ok and score.coverage_ok
            if score.reason:
                reasons.append(f"{task}: {score.reason}")

        overall = self._weighted_task_mean(task_scores, self._score_type_weights())
        return (
            StratifiedScore(
                overall=overall if overall is not None else 0.0,
                per_band=per_band,
                coverage_ok=coverage_ok,
                reason="; ".join(reasons) or None,
            ),
            task_scores,
        )

    def _scored_result(
        self,
        miner_id: str,
        score: StratifiedScore,
        rollouts: list[RolloutResult],
        task_scores: dict[str, float] | None = None,
    ) -> MinerEpochResult:
        """Attach dashboard telemetry without changing the scoring result."""
        by_task: dict[str, list[RolloutResult]] = defaultdict(list)
        for rollout in rollouts:
            by_task[self._rollout_task(rollout)].append(rollout)

        def signed_correctness(values: list[bool]) -> float | None:
            return (
                sum(1.0 if bool(verified) else -1.0 for verified in values)
                / len(values)
                if values
                else None
            )

        def mean(values: list[int | float]) -> float | None:
            return sum(float(value) for value in values) / len(values) if values else None

        task_completion_len: dict[str, float] = {}
        task_correctness_score: dict[str, float] = {}
        task_original_score: dict[str, float] = {}
        task_original_correctness_score: dict[str, float] = {}
        task_original_completion_len: dict[str, float] = {}
        for task, task_rollouts in by_task.items():
            miner_lengths = [
                result.miner_completion_len
                for result in task_rollouts
                if result.miner_completion_len is not None
            ]
            if (value := mean(miner_lengths)) is not None:
                task_completion_len[task] = value

            original_lengths = [
                result.original_completion_len
                for result in task_rollouts
                if result.original_completion_len is not None
            ]
            if (value := mean(original_lengths)) is not None:
                task_original_completion_len[task] = value

            miner_verified = [
                result.miner_verified
                for result in task_rollouts
                if result.miner_verified is not None
            ]
            if (value := signed_correctness(miner_verified)) is not None:
                task_correctness_score[task] = value

            original_verified = [
                result.original_verified
                for result in task_rollouts
                if result.original_verified is not None
            ]
            if original_verified:
                task_original_score[task] = sum(bool(v) for v in original_verified) / len(
                    original_verified
                )
            if (value := signed_correctness(original_verified)) is not None:
                task_original_correctness_score[task] = value

        weights = self._score_type_weights()
        return MinerEpochResult(
            miner_id=miner_id,
            score=score,
            completion_len=self._weighted_task_mean(task_completion_len, weights),
            correctness_score=self._weighted_task_mean(task_correctness_score, weights),
            original_score=self._weighted_task_mean(task_original_score, weights),
            original_correctness_score=self._weighted_task_mean(
                task_original_correctness_score, weights
            ),
            original_completion_len=self._weighted_task_mean(
                task_original_completion_len, weights
            ),
            telemetry_count=len(rollouts),
            task_scores=dict(task_scores or {}),
            task_completion_len=task_completion_len,
            task_correctness_score=task_correctness_score,
            task_original_score=task_original_score,
            task_original_correctness_score=task_original_correctness_score,
            task_original_completion_len=task_original_completion_len,
        )

    @staticmethod
    def _table_cell(text: object, *, limit: int = 80) -> str:
        value = str(text)
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)].rstrip() + "..."

    @classmethod
    def _results_table_lines(
        cls,
        results: dict[str, MinerEpochResult],
        *,
        full_eval_miners: set[str] | None = None,
    ) -> list[str]:
        if not results:
            return ["no miner results"]

        def stage_for(miner_id: str, result: MinerEpochResult) -> str:
            if result.rejected_reason == "skipped_stagnant_submission":
                return "skipped"
            if full_eval_miners is None:
                return "full"
            if miner_id in full_eval_miners:
                return "full"
            return "qualification-only"

        def sort_key(item: tuple[str, MinerEpochResult]):
            miner_id, result = item
            score = result.score.overall if result.score is not None else float("-inf")
            coverage_ok = result.score.coverage_ok if result.score is not None else False
            return (result.score is None, not coverage_ok, -score, miner_id)

        def component_score(result: MinerEpochResult, task: str) -> str:
            value = result.task_scores.get(task)
            return "-" if value is None else f"{value:.4f}"

        rows: list[list[str]] = []
        rank = 0
        for miner_id, result in sorted(results.items(), key=sort_key):
            if result.score is None:
                score = "-"
                math_score = "-"
                long_qa_score = "-"
                science_score = "-"
                coverage = "-"
                reason = result.rejected_reason or "rejected"
                bands = "-"
                rank_text = "-"
            else:
                rank += 1
                score = f"{result.score.overall:.4f}"
                math_score = component_score(result, TASK_MATH)
                long_qa_score = component_score(result, TASK_LONG_CONTEXT_QA)
                science_score = component_score(result, TASK_MULTIPLE_CHOICE)
                coverage = "ok" if result.score.coverage_ok else "missing"
                reason = result.score.reason or "ok"
                bands = ", ".join(
                    f"{band}={value:.4f}"
                    for band, value in sorted(result.score.per_band.items())
                ) or "-"
                rank_text = str(rank)
            rows.append(
                [
                    rank_text,
                    miner_id,
                    stage_for(miner_id, result),
                    score,
                    math_score,
                    long_qa_score,
                    science_score,
                    coverage,
                    cls._table_cell(reason, limit=64),
                    cls._table_cell(bands, limit=96),
                ]
            )

        headers = [
            "rank",
            "miner",
            "stage",
            "score",
            "math",
            "long_qa",
            "science",
            "coverage",
            "reason",
            "bands",
        ]
        widths = [
            max(len(headers[idx]), *(len(row[idx]) for row in rows))
            for idx in range(len(headers))
        ]
        separator = "-+-".join("-" * width for width in widths)

        def format_row(values: list[str]) -> str:
            return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

        return [format_row(headers), separator, *(format_row(row) for row in rows)]

    def _log_results_table(
        self,
        results: dict[str, MinerEpochResult],
        *,
        full_eval_miners: set[str] | None = None,
    ) -> None:
        self._log("evaluation summary:")
        for line in self._results_table_lines(results, full_eval_miners=full_eval_miners):
            self._log(line)

    def _log_sample_result(
        self,
        label: str,
        *,
        token_count: int | None = None,
        verified: bool | None = None,
        search_used: bool = False,
    ) -> None:
        details: list[str] = []
        if token_count is not None:
            details.append(f"tokens={token_count}")
        if verified is not None:
            details.append(f"verified={verified}")
        if search_used:
            details.append("search_used=true")
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"[thinker-validator] sample {label} result{suffix}", flush=True)

    def _log_test_long_context_samples(
        self,
        batch: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
        results_by_miner: dict[str, list[LongContextMinerResult]],
        errors_by_miner: dict[str, str],
        *,
        limit: int = 5,
    ) -> None:
        sample_count = min(max(0, int(limit)), len(batch))
        self._log(f"test mode long_qa: logging first {sample_count} sample(s)")
        miner_ids = sorted(set(results_by_miner) | set(errors_by_miner))
        for index in range(sample_count):
            instance = batch[index]
            payload: dict[str, Any] = {
                "sample": index + 1,
                "seed": instance.seed,
                "question": instance.question,
                "gold_answer": instance.gold_answer,
                "gold_document_ids": [
                    instance.seed_hits[index - 1].document.doc_id
                    for index in instance.supporting_document_indices
                ],
                "miners": {},
            }
            if index < len(originals):
                original = originals[index]
                payload["baseline"] = {
                    "answer": original.text,
                    "verified": original.verified,
                    "tokens": original.completion_len,
                }

            miner_payloads: dict[str, Any] = {}
            for miner_id in miner_ids:
                if miner_id in errors_by_miner:
                    miner_payloads[miner_id] = {"error": errors_by_miner[miner_id]}
                    continue
                miner_results = results_by_miner[miner_id]
                if index >= len(miner_results):
                    miner_payloads[miner_id] = {"error": "missing sample result"}
                    continue
                result = miner_results[index]
                miner_payloads[miner_id] = {
                    "search_query": result.search_query,
                    "selected_indices": list(result.selected_document_indices),
                    "tokens": result.miner.completion_len,
                    "verified": result.miner.verified,
                    "reward": result.score,
                }
            payload["miners"] = miner_payloads
            self._log(
                "test mode long_qa sample "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )

    def _log_test_long_context_generated_samples(
        self,
        batch: list[LongContextQAInstance],
        *,
        limit: int = 5,
    ) -> None:
        sample_count = min(max(0, int(limit)), len(batch))
        self._log(
            f"test mode long_qa: generated batch; logging first {sample_count} sample(s)"
        )
        for sample_index, instance in enumerate(batch[:sample_count], start=1):
            supporting_documents = [
                instance.seed_hits[index - 1].document
                for index in instance.supporting_document_indices
            ]
            payload = {
                "sample": sample_index,
                "seed": instance.seed,
                "question": instance.question,
                "gold_answer": instance.gold_answer,
                "supporting_documents": [
                    {
                        "id": document.doc_id,
                        "title": document.title,
                        "text_preview": (document.text or document.contents)[:500],
                    }
                    for document in supporting_documents
                ],
            }
            self._log(
                "test mode long_qa generated sample "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )

    @staticmethod
    def _math_gold_answer(instance: ProblemInstance) -> str | None:
        track = get_track(instance.track)
        dataset_and_item = getattr(track, "_dataset_and_item", None)
        if callable(dataset_and_item):
            try:
                _name, _dataset, item = dataset_and_item(instance.seed)
                if isinstance(item, dict) and "answer" in item:
                    return str(item["answer"])
            except Exception:
                pass

        make_instance = getattr(track, "_instance", None)
        if not callable(make_instance):
            return None
        try:
            track_instance = make_instance(instance.seed)
        except Exception:
            return None

        get_solution = getattr(track_instance, "get_solution", None)
        if callable(get_solution):
            try:
                return str(get_solution())
            except Exception:
                pass
        for attribute in ("gold_answer", "answer", "solution"):
            if not hasattr(track_instance, attribute):
                continue
            value = getattr(track_instance, attribute)
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            return str(value)
        return None

    def _log_test_math_samples(
        self,
        batch: list[ProblemInstance],
        original_rollouts: list[OriginalRollout],
        completions_by_miner: dict[str, list[tuple[str, int]]],
        scores_by_miner: dict[str, list[float]],
        errors_by_miner: dict[str, str],
        miner_ids: list[str],
    ) -> None:
        if not batch:
            self._log("test mode math sample: no math problem available")
            return
        self._log(
            "test mode math: logging first verified-correct sample for each miner"
        )

        for miner_id in miner_ids:
            if miner_id in errors_by_miner:
                self._log(
                    "test mode math sample "
                    + json.dumps(
                        {"miner": miner_id, "error": errors_by_miner[miner_id]},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                continue

            completions = completions_by_miner.get(miner_id, [])
            selected: tuple[int, str, int] | None = None
            for index, instance in enumerate(batch):
                if index >= len(completions):
                    continue
                completion, token_count = completions[index]
                if get_track(instance.track).verify(instance.seed, completion):
                    selected = (index, completion, token_count)
                    break

            if selected is None:
                self._log(
                    "test mode math sample "
                    + json.dumps(
                        {
                            "miner": miner_id,
                            "error": "no verified math sample",
                            "samples_checked": min(len(batch), len(completions)),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                continue

            index, completion, token_count = selected
            instance = batch[index]
            original = (
                original_rollouts[index] if index < len(original_rollouts) else None
            )
            payload: dict[str, Any] = {
                "sample": index + 1,
                "seed": instance.seed,
                "track": instance.track,
                "problem": instance.prompt,
                "gold_answer": self._math_gold_answer(instance),
                "miner": miner_id,
            }
            if original is not None:
                payload["baseline"] = {
                    "verified": original.verified,
                    "tokens": original.completion_len,
                }
            payload["miner_response"] = completion
            payload["tokens"] = token_count
            payload["verified"] = True
            scores = scores_by_miner.get(miner_id)
            if scores is not None and index < len(scores):
                payload["reward"] = scores[index]
            self._log(
                "test mode math sample "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )

    def _log_test_science_samples(
        self,
        batch: list[MultipleChoiceInstance],
        originals: list[MultipleChoiceAnswer],
        results_by_miner: dict[str, list[MultipleChoiceMinerResult]],
        errors_by_miner: dict[str, str],
        miner_ids: list[str],
    ) -> None:
        if not batch:
            self._log("test mode science sample: no science problem available")
            return
        self._log(
            "test mode science: logging first verified-correct sample for each miner"
        )

        for miner_id in miner_ids:
            if miner_id in errors_by_miner:
                self._log(
                    "test mode science sample "
                    + json.dumps(
                        {"miner": miner_id, "error": errors_by_miner[miner_id]},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                continue

            miner_results = results_by_miner.get(miner_id, [])
            selected_index = next(
                (
                    index
                    for index, result in enumerate(miner_results)
                    if index < len(batch) and result.miner.verified
                ),
                None,
            )
            if selected_index is None:
                self._log(
                    "test mode science sample "
                    + json.dumps(
                        {
                            "miner": miner_id,
                            "error": "no verified science sample",
                            "samples_checked": min(len(batch), len(miner_results)),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                continue

            instance = batch[selected_index]
            result = miner_results[selected_index]
            original = originals[selected_index] if selected_index < len(originals) else None
            payload: dict[str, Any] = {
                "sample": selected_index + 1,
                "seed": instance.seed,
                "track": TASK_MULTIPLE_CHOICE,
                "problem_id": instance.problem_id,
                "problem": instance.prompt,
                "gold_answer": instance.ground_truth,
                "enable_thinking": instance.enable_thinking,
                "miner": miner_id,
                "miner_response": result.miner.response,
                "parsed_answer": result.miner.text,
                "tokens": result.miner.completion_len,
                "verified": True,
                "reward": result.score,
            }
            if original is not None:
                payload["baseline"] = {
                    "verified": original.verified,
                    "tokens": original.completion_len,
                }
            self._log(
                "test mode science sample "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )

    def build_batch(
        self,
        n_problems: int | None = None,
        *,
        epoch: int = 0,
        common_seed: str | None = None,
        seed_namespace: str = "full_evaluation",
    ) -> list[ProblemInstance]:
        return self.build_math_batch(
            n_problems,
            epoch=epoch,
            common_seed=common_seed,
            seed_namespace=seed_namespace,
        )

    @staticmethod
    def _track_for_seed(tracks: list[str], seed: str) -> str:
        index = int.from_bytes(hashlib.sha256(seed.encode("ascii")).digest()[:8], "big")
        return tracks[index % len(tracks)]

    @staticmethod
    def _retry_private_seed(
        private_seed: str, *, epoch: int, namespace: str, attempt: int
    ) -> str:
        return hashlib.sha256(
            b"thinker-private-sample-retry-v1\0"
            + private_seed.encode("utf-8")
            + b"\0"
            + str(epoch).encode("ascii")
            + b"\0"
            + namespace.encode("utf-8")
            + b"\0"
            + str(attempt).encode("ascii")
        ).hexdigest()

    def build_math_batch(
        self,
        n_problems: int | None = None,
        *,
        epoch: int = 0,
        common_seed: str | None = None,
        seed_namespace: str = "full_evaluation",
    ) -> list[ProblemInstance]:
        tracks = sorted(registered_tracks())
        if not tracks:
            raise RuntimeError("no problem tracks registered")
        target = n_problems if n_problems is not None else self._config.n_problems_per_epoch
        namespace = f"{seed_namespace}:math"
        private_seed = self._seed_fn()
        plan = build_sample_seed_plan(
            target,
            private_seed=private_seed,
            epoch=epoch,
            namespace=namespace,
            common_seed=common_seed,
        )
        batch: list[ProblemInstance] = []
        attempt = 0
        skipped = 0
        max_attempts = max(target * 50, 50)
        with self._progress(target, "math batch", unit="sample") as progress:
            for seed in plan.seeds[: plan.common_count]:
                track = self._track_for_seed(tracks, seed)
                # Common seeds align the selected track and source row. A track
                # may still apply validator-local rendering to that shared row.
                try:
                    instance = render_instance(track, seed)
                    # A valid owner seed takes precedence over validator-local history;
                    # otherwise one validator's private history could silently remove a
                    # supposedly common problem. The epoch-bound derivation makes repeats
                    # across owner rounds cryptographically unlikely.
                    self._decontam.check_and_record(track, instance.prompt)
                except Exception as exc:
                    skipped += 1
                    self._log(
                        f"WARNING: math batch: skipping common sample "
                        f"(track={track!r}) after render failure: "
                        f"{type(exc).__name__}"
                    )
                    continue
                batch.append(instance)
                progress.update(1)
            private_candidates = list(plan.seeds[plan.common_count :])
            while len(batch) < target and attempt < max_attempts:
                attempt += 1
                if private_candidates:
                    seed = private_candidates.pop(0)
                else:
                    seed = self._retry_private_seed(
                        private_seed,
                        epoch=epoch,
                        namespace=namespace,
                        attempt=attempt,
                    )
                track = self._track_for_seed(tracks, seed)
                try:
                    instance = render_instance(track, seed)
                except Exception as exc:
                    skipped += 1
                    self._log(
                        f"WARNING: math batch: skipping private sample "
                        f"(track={track!r}) after render failure: "
                        f"{type(exc).__name__}"
                    )
                    progress.set_postfix(attempts=attempt, refresh=False)
                    continue
                if self._decontam.check_and_record(track, instance.prompt):
                    batch.append(instance)
                    progress.update(1)
                    progress.set_postfix(attempts=attempt, refresh=False)
        if skipped:
            self._log(
                f"math batch: skipped {skipped} sample(s) due to render failures "
                f"(e.g. transient dataset/network errors)"
            )
        if len(batch) < target:
            raise RuntimeError(
                f"could not fill batch ({len(batch)}/{target}) without repeats in {max_attempts} "
                "attempts -- bank too small for n_problems_per_epoch"
            )
        return batch

    def build_long_context_batch(
        self,
        n_questions: int | None = None,
        *,
        epoch: int = 0,
        common_seed: str | None = None,
        seed_namespace: str = "full_evaluation",
    ) -> list[LongContextQAInstance]:
        target = (
            n_questions
            if n_questions is not None
            else self._config.n_long_context_qa_per_epoch
        )
        if target <= 0:
            return []
        if self._long_context is None:
            raise RuntimeError("long-context QA requires a retrieval-backed evaluator")
        self._log(f"long-context QA batch: building {target} sample(s)")
        plan = build_sample_seed_plan(
            target,
            private_seed=self._seed_fn(),
            epoch=epoch,
            namespace=f"{seed_namespace}:long_context_qa",
            common_seed=common_seed,
        )
        batch = self._long_context.generate_instances(list(plan.seeds))
        self._log(f"long-context QA batch: complete {len(batch)}/{target}")
        return batch

    def build_epoch_batch(
        self,
        n_math: int | None = None,
        n_long_context_qa: int | None = None,
        *,
        epoch: int = 0,
        common_seed: str | None = None,
        seed_namespace: str = "full_evaluation",
    ) -> EpochBatch:
        return EpochBatch(
            math=self.build_math_batch(
                n_math,
                epoch=epoch,
                common_seed=common_seed,
                seed_namespace=seed_namespace,
            ),
            long_context_qa=self.build_long_context_batch(
                n_long_context_qa,
                epoch=epoch,
                common_seed=common_seed,
                seed_namespace=seed_namespace,
            ),
        )

    def build_multiple_choice_batch(
        self,
        n_samples: int | None = None,
        n_thinking_samples: int | None = None,
        *,
        epoch: int = 0,
        common_seed: str | None = None,
        seed_namespace: str = "qualification",
    ) -> list[MultipleChoiceInstance]:
        target = max(0, int(n_samples or 0))
        if target == 0:
            return []
        if self._multiple_choice is None:
            raise RuntimeError("multiple-choice qualification requires an evaluator")
        self._log(f"multiple-choice qualification batch: building {target} sample(s)")
        plan = build_sample_seed_plan(
            target,
            private_seed=self._seed_fn(),
            epoch=epoch,
            namespace=f"{seed_namespace}:multiple_choice",
            common_seed=common_seed,
        )
        batch = self._multiple_choice.generate_instances(
            plan.seeds,
            thinking_samples=n_thinking_samples,
            common_prefix_count=plan.common_count,
        )
        self._log(f"multiple-choice qualification batch: complete {len(batch)}/{target}")
        return batch

    def _fetch_and_decrypt(self, pointer: MinerSubmissionPointer) -> bytes:
        submission = self._transport.fetch(pointer)
        if content_hash(submission) != pointer.sha256:
            raise ValueError("hash_mismatch")
        max_ciphertext = max_encrypted_adapter_ciphertext_bytes(self._config.max_adapter_bytes)
        if len(submission.ciphertext) > max_ciphertext:
            raise ValueError("encrypted_submission_too_large")
        if len(submission.wrapped_keys) > self._config.max_submission_recipients:
            raise ValueError("too_many_submission_recipients")
        return decrypt_as_recipient(submission, self._recipient_id, self._recipient_privkey)

    @staticmethod
    def _adapter_hash(adapter_files: dict[str, bytes]) -> str:
        digest = hashlib.sha256()
        for name in sorted(adapter_files):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(adapter_files[name])
            digest.update(b"\0")
        return digest.hexdigest()

    def _dedupe_adapter_file_miners(
        self,
        miners: list[tuple[str, dict[str, bytes]]],
        *,
        label: str,
    ) -> tuple[list[tuple[str, dict[str, bytes]]], dict[str, str]]:
        unique: list[tuple[str, dict[str, bytes]]] = []
        canonical_by_hash: dict[str, str] = {}
        aliases: dict[str, str] = {}
        for miner_id, adapter_files in miners:
            adapter_hash = self._adapter_hash(adapter_files)
            canonical = canonical_by_hash.get(adapter_hash)
            if canonical is None:
                canonical_by_hash[adapter_hash] = miner_id
                unique.append((miner_id, adapter_files))
            else:
                aliases[miner_id] = canonical
        if aliases:
            alias_text = ", ".join(
                f"{alias}->{canonical}" for alias, canonical in aliases.items()
            )
            self._log(
                f"{label}: reusing canonical results for duplicate adapter(s): "
                f"{alias_text}"
            )
        return unique, aliases

    def _dedupe_prepared_miners(
        self,
        miners: list[tuple[str, PreparedMinerSubmission]],
        *,
        label: str,
    ) -> tuple[list[tuple[str, PreparedMinerSubmission]], dict[str, str]]:
        unique: list[tuple[str, PreparedMinerSubmission]] = []
        canonical_by_hash: dict[str, str] = {}
        aliases: dict[str, str] = {}
        for miner_id, prepared in miners:
            canonical = canonical_by_hash.get(prepared.adapter_hash)
            if canonical is None:
                canonical_by_hash[prepared.adapter_hash] = miner_id
                unique.append((miner_id, prepared))
            else:
                aliases[miner_id] = canonical
        if aliases:
            alias_text = ", ".join(
                f"{alias}->{canonical}" for alias, canonical in aliases.items()
            )
            self._log(
                f"{label}: reusing canonical results for duplicate adapter(s): "
                f"{alias_text}"
            )
        return unique, aliases

    @staticmethod
    def _alias_duplicate_adapter_outcomes(
        results: dict[str, Any],
        errors: dict[str, str],
        aliases: dict[str, str],
    ) -> None:
        for alias, canonical in aliases.items():
            if canonical in errors:
                errors[alias] = errors[canonical]
            elif canonical in results:
                results[alias] = results[canonical]

    def _check_fingerprint(self, miner_id: str, adapter: ValidatedAdapter) -> str | None:
        if miner_id in self._fingerprint_exempt_miner_ids:
            return None
        fp = compute_lora_fingerprint(adapter.state_dict, adapter.config)
        for other_id, other_fp in self._fingerprints.items():
            if other_id == miner_id:
                continue
            collision, reason = fingerprints_collide(fp, other_fp)
            if collision:
                return f"fingerprint_collision vs {other_id}: {reason}"
        self._fingerprints[miner_id] = fp
        return None

    @staticmethod
    def _validate_completion_count(
        source: str, completions: list[tuple[str, int]], expected: int
    ) -> list[tuple[str, int]]:
        if len(completions) != expected:
            raise ValueError(
                f"{source} returned {len(completions)} completions for {expected} prompts"
            )
        return completions

    @staticmethod
    def _generation_budget(completion_len: int) -> int:
        return max(1, int(completion_len))

    def _generate_miner_with_budgets(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        budgets: list[int],
        *,
        progress_label: str | None = None,
    ) -> list[tuple[str, int]]:
        if len(prompts) != len(budgets):
            raise ValueError("prompts and budgets must have the same length")
        if not prompts:
            return []

        grouped: dict[int, list[int]] = {}
        for index, budget in enumerate(budgets):
            grouped.setdefault(self._generation_budget(budget), []).append(index)

        completions: list[tuple[str, int] | None] = [None] * len(prompts)
        label = progress_label or f"miner {miner_id} math"
        with self._progress(
            len(prompts), label, unit="prompt", suppress_inference=True
        ) as progress:
            for budget, indexes in grouped.items():
                chunk_prompts = [prompts[index] for index in indexes]
                chunk_completions = self._validate_completion_count(
                    miner_id,
                    self._inference.generate_limited(
                        miner_id,
                        adapter_files,
                        chunk_prompts,
                        max_new_tokens=budget,
                        system_prompt=MATH_SYSTEM_PROMPT,
                    ),
                    len(indexes),
                )
                for index, completion in zip(indexes, chunk_completions):
                    completions[index] = completion
                progress.update(len(indexes))

        if any(completion is None for completion in completions):
            raise ValueError("miner budgeted generation left a missing completion")
        return [completion for completion in completions if completion is not None]

    def _generate_miners_math_completions_batch(
        self,
        miners: list[tuple[str, dict[str, bytes]]],
        batch: list[ProblemInstance],
        original_rollouts: list[OriginalRollout],
    ) -> tuple[dict[str, list[tuple[str, int]]], dict[str, str]]:
        """Generate math completions for many miners at once.

        Per-problem budgets only depend on `original_rollouts` (the shared
        baseline), not on which miner is answering, so every miner in this
        batch can share one flat request list scored against one vLLM call.
        Falls back to one call per miner -- identical to the prior behavior --
        if the batched call itself fails for any reason.

        Returns (completions_by_miner, errors_by_miner) -- a miner with a
        generation failure shows up only in `errors_by_miner` so one miner's
        broken adapter can't take down the whole batch.
        """
        completions: dict[str, list[tuple[str, int]]] = {}
        errors: dict[str, str] = {}
        if not miners:
            return completions, errors
        if not batch:
            return {miner_id: [] for miner_id, _adapter_files in miners}, errors

        unique_miners, aliases = self._dedupe_adapter_file_miners(
            miners, label="math"
        )
        prompts = [instance.prompt for instance in batch]
        budgets = [self._generation_budget(o.completion_len) for o in original_rollouts]

        requests: list[tuple[str, dict[str, bytes], str]] = []
        max_new_tokens_list: list[int] = []
        for miner_id, adapter_files in unique_miners:
            for prompt, budget in zip(prompts, budgets):
                requests.append((miner_id, adapter_files, prompt))
                max_new_tokens_list.append(budget)
        try:
            self._log(
                f"math: batched generation for {len(unique_miners)} unique adapter(s) x "
                f"{len(prompts)} problem(s) = {len(requests)} request(s) in one call"
            )
            with self._progress(
                len(requests),
                "miner math generation",
                unit="prompt",
                suppress_inference=True,
            ) as progress:
                flat_completions = self._inference.generate_for_miners_batch(
                    requests,
                    max_new_tokens_list=max_new_tokens_list,
                    system_prompt=MATH_SYSTEM_PROMPT,
                )
                progress.update(len(requests))
            if len(flat_completions) != len(requests):
                raise ValueError(
                    f"batched math generation returned {len(flat_completions)} "
                    f"completion(s) for {len(requests)} request(s)"
                )
            cursor = 0
            for miner_id, _adapter_files in unique_miners:
                completions[miner_id] = list(flat_completions[cursor : cursor + len(prompts)])
                cursor += len(prompts)
            self._alias_duplicate_adapter_outcomes(completions, errors, aliases)
            return completions, errors
        except Exception as exc:
            self._report_batched_math_failure(exc)

        for miner_id, adapter_files in unique_miners:
            try:
                completions[miner_id] = self._generate_miner_with_budgets(
                    miner_id, adapter_files, prompts, budgets
                )
            except Exception as exc:
                errors[miner_id] = str(exc)
        self._alias_duplicate_adapter_outcomes(completions, errors, aliases)
        return completions, errors

    def _score_miners_math_completions_batch(
        self,
        batch: list[ProblemInstance],
        original_rollouts: list[OriginalRollout],
        completions_by_miner: dict[str, list[tuple[str, int]]],
    ) -> dict[str, list[float]]:
        if len(original_rollouts) != len(batch):
            raise ValueError(
                f"baseline_mismatch: {len(original_rollouts)} rollouts for "
                f"{len(batch)} problems"
            )
        scores_by_miner: dict[str, list[float]] = {
            miner_id: [0.0] * len(batch) for miner_id in completions_by_miner
        }
        miner_ids = list(completions_by_miner)
        for miner_id, completions in completions_by_miner.items():
            if len(completions) != len(batch):
                raise ValueError(
                    f"math returned {len(completions)} completion(s) for "
                    f"{miner_id}; expected {len(batch)}"
                )

        for index, (instance, original) in enumerate(zip(batch, original_rollouts)):
            track = get_track(instance.track)
            verified_by_miner: dict[str, bool] = {}
            lens_by_miner: dict[str, int] = {}
            for miner_id in miner_ids:
                completion, token_count = completions_by_miner[miner_id][index]
                verified_by_miner[miner_id] = track.verify(instance.seed, completion)
                lens_by_miner[miner_id] = token_count

            base_rewards = [
                relative_reasoning_reward(
                    original_verified=original.verified,
                    miner_verified=verified_by_miner[miner_id],
                    original_completion_len=original.completion_len,
                    miner_completion_len=lens_by_miner[miner_id],
                )
                for miner_id in miner_ids
            ]
            rewards = peer_completion_efficiency_rewards(
                original_verified=original.verified,
                miner_verified=[
                    verified_by_miner[miner_id] for miner_id in miner_ids
                ],
                miner_completion_lens=[
                    lens_by_miner[miner_id] for miner_id in miner_ids
                ],
                base_rewards=base_rewards,
            )
            for miner_id, reward in zip(miner_ids, rewards):
                scores_by_miner[miner_id][index] = reward

        return scores_by_miner

    def _report_batched_math_failure(self, exc: Exception) -> None:
        """Falling back to the slow per-miner path is meant to keep the
        epoch running, not to hide that the fast path is broken. Print the
        full traceback the first time this happens per process so the real
        root cause is visible immediately instead of buried under hundreds
        of vLLM progress-bar lines; later occurrences in the same run still
        get a one-line, clearly-tagged reminder rather than going silent.
        """
        if not self._batched_math_failure_logged:
            self._batched_math_failure_logged = True
            print(
                "[thinker-validator] ERROR: cross-miner batched math generation "
                "failed -- falling back to the slow per-miner/per-problem path "
                "for the rest of this run. This is NOT expected to happen in "
                "normal operation; fix the root cause below rather than relying "
                "on the fallback.\n"
                + traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[thinker-validator] ERROR (repeat, see earlier traceback): "
                f"batched math generation failed again ({exc})",
                file=sys.stderr,
                flush=True,
            )

    def _report_batched_task_failure(
        self, label: str, exc: Exception, *, already_logged: bool
    ) -> bool:
        if not already_logged:
            print(
                f"[thinker-validator] ERROR: cross-miner batched {label} failed -- "
                "falling back to the per-miner path. This is NOT expected in "
                "normal operation; fix the root cause below rather than relying "
                "on the fallback.\n"
                + traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
            return True
        print(
            f"[thinker-validator] ERROR (repeat, see earlier traceback): "
            f"batched {label} failed again ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return True

    def _score_miners_long_context_batch(
        self,
        miners: list[tuple[str, PreparedMinerSubmission]],
        batch: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
    ) -> tuple[dict[str, list[LongContextMinerResult]], dict[str, str]]:
        results: dict[str, list[LongContextMinerResult]] = {}
        errors: dict[str, str] = {}
        if not miners:
            return results, errors
        if not batch:
            return {miner_id: [] for miner_id, _prepared in miners}, errors
        if self._long_context is None:
            return {}, {
                miner_id: "long_context_unavailable: evaluator missing"
                for miner_id, _prepared in miners
            }

        unique_miners, aliases = self._dedupe_prepared_miners(
            miners, label="long-context QA"
        )
        try:
            self._log(
                f"long-context QA: batched scoring for {len(unique_miners)} unique adapter(s) x "
                f"{len(batch)} question(s)"
            )
            results = self._long_context.score_miners_batch(
                miners=[
                    (miner_id, prepared.adapter_files)
                    for miner_id, prepared in unique_miners
                ],
                instances=batch,
                originals=originals,
            )
            for miner_id, _prepared in unique_miners:
                scored = results.get(miner_id)
                if scored is None or len(scored) != len(batch):
                    got = "missing" if scored is None else str(len(scored))
                    raise ValueError(
                        f"batched long-context QA returned {got} result(s) "
                        f"for {miner_id}; expected {len(batch)}"
                    )
            self._alias_duplicate_adapter_outcomes(results, errors, aliases)
            return results, errors
        except Exception as exc:
            self._batched_long_context_failure_logged = self._report_batched_task_failure(
                "long-context QA",
                exc,
                already_logged=self._batched_long_context_failure_logged,
            )

        for miner_id, prepared in unique_miners:
            try:
                scored = self._long_context.score_miner_batch(
                    miner_id=miner_id,
                    adapter_files=prepared.adapter_files,
                    instances=batch,
                    originals=originals,
                )
                if len(scored) != len(batch):
                    raise ValueError(
                        f"long-context QA returned {len(scored)} result(s) for "
                        f"{len(batch)} question(s)"
                    )
                results[miner_id] = scored
            except Exception as exc:
                errors[miner_id] = str(exc)
        if results:
            results = self._long_context.apply_peer_efficiency(results)
        self._alias_duplicate_adapter_outcomes(results, errors, aliases)
        return results, errors

    def _score_miners_multiple_choice_batch(
        self,
        miners: list[tuple[str, PreparedMinerSubmission]],
        batch: list[MultipleChoiceInstance],
        originals: list[MultipleChoiceAnswer],
    ) -> tuple[dict[str, list[MultipleChoiceMinerResult]], dict[str, str]]:
        results: dict[str, list[MultipleChoiceMinerResult]] = {}
        errors: dict[str, str] = {}
        if not miners:
            return results, errors
        if not batch:
            return {miner_id: [] for miner_id, _prepared in miners}, errors
        if self._multiple_choice is None:
            return {}, {
                miner_id: "multiple_choice_unavailable: evaluator missing"
                for miner_id, _prepared in miners
            }

        unique_miners, aliases = self._dedupe_prepared_miners(
            miners, label="multiple-choice"
        )
        try:
            self._log(
                f"multiple-choice: batched scoring for {len(unique_miners)} unique adapter(s) x "
                f"{len(batch)} sample(s)"
            )
            results = self._multiple_choice.score_miners_batch(
                miners=[
                    (miner_id, prepared.adapter_files)
                    for miner_id, prepared in unique_miners
                ],
                instances=batch,
                originals=originals,
            )
            for miner_id, _prepared in unique_miners:
                scored = results.get(miner_id)
                if scored is None or len(scored) != len(batch):
                    got = "missing" if scored is None else str(len(scored))
                    raise ValueError(
                        f"batched multiple-choice returned {got} result(s) "
                        f"for {miner_id}; expected {len(batch)}"
                    )
            self._alias_duplicate_adapter_outcomes(results, errors, aliases)
            return results, errors
        except Exception as exc:
            self._batched_multiple_choice_failure_logged = self._report_batched_task_failure(
                "multiple-choice",
                exc,
                already_logged=self._batched_multiple_choice_failure_logged,
            )

        for miner_id, prepared in unique_miners:
            try:
                scored = self._multiple_choice.score_miner_batch(
                    miner_id=miner_id,
                    adapter_files=prepared.adapter_files,
                    instances=batch,
                    originals=originals,
                )
                if len(scored) != len(batch):
                    raise ValueError(
                        f"multiple-choice returned {len(scored)} result(s) for "
                        f"{len(batch)} sample(s)"
                    )
                results[miner_id] = scored
            except Exception as exc:
                errors[miner_id] = str(exc)
        self._alias_duplicate_adapter_outcomes(results, errors, aliases)
        return results, errors

    def score_original_batch(self, batch: list[ProblemInstance]) -> list[OriginalRollout]:
        if not batch:
            return []
        prompts = [instance.prompt for instance in batch]
        with self._progress(
            len(prompts),
            "baseline math",
            unit="prompt",
            suppress_inference=True,
        ) as progress:
            completions = self._validate_completion_count(
                "original_model",
                self._inference.generate_original_limited(
                    prompts,
                    max_new_tokens=None,
                    enable_thinking=True,
                    system_prompt=MATH_SYSTEM_PROMPT,
                ),
                len(batch),
            )
            progress.update(len(prompts))
        if completions:
            self._log_sample_result(
                "original math",
                token_count=completions[0][1],
            )
        rollouts: list[OriginalRollout] = []
        for instance, (completion, token_count) in zip(batch, completions):
            verified = get_track(instance.track).verify(instance.seed, completion)
            rollouts.append(OriginalRollout(verified=verified, completion_len=token_count))
        return rollouts

    def score_original_long_context_batch(
        self, batch: list[LongContextQAInstance]
    ) -> list[LongContextAnswer]:
        if not batch:
            return []
        if self._long_context is None:
            raise RuntimeError("long-context QA requires a retrieval-backed evaluator")
        self._log(f"original long-context QA: scoring {len(batch)} sample(s)")
        answers = self._long_context.score_original_batch(batch)
        self._log(f"original long-context QA: complete {len(answers)}/{len(batch)}")
        if answers:
            self._log_sample_result(
                "original long-context QA",
                token_count=answers[0].completion_len,
                verified=answers[0].verified,
            )
        return answers

    def score_original_multiple_choice_batch(
        self, batch: list[MultipleChoiceInstance]
    ) -> list[MultipleChoiceAnswer]:
        if not batch:
            return []
        if self._multiple_choice is None:
            raise RuntimeError("multiple-choice qualification requires an evaluator")
        self._log(f"original multiple-choice: scoring {len(batch)} sample(s)")
        answers = self._multiple_choice.score_original_batch(batch)
        self._log(f"original multiple-choice: complete {len(answers)}/{len(batch)}")
        if answers:
            self._log_sample_result(
                "original multiple-choice",
                token_count=answers[0].completion_len,
                verified=answers[0].verified,
            )
        return answers

    def prepare_miner(
        self,
        miner_id: str,
        pointer: MinerSubmissionPointer,
    ) -> PreparedMinerSubmission | MinerEpochResult:
        try:
            plaintext = self._fetch_and_decrypt(pointer)
        except Exception as exc:
            return MinerEpochResult(miner_id, None, f"fetch_or_decrypt_failed: {exc}")

        try:
            adapter_files = unpack_adapter_bundle(
                plaintext,
                max_total_bytes=self._config.max_adapter_bytes,
                max_config_bytes=self._config.max_adapter_config_bytes,
            )
            adapter = validate_adapter_files(adapter_files, self._config)
        except Exception as exc:
            return MinerEpochResult(miner_id, None, f"malformed_bundle: {exc}")

        copy_reason = self._check_fingerprint(miner_id, adapter)
        if copy_reason is not None:
            return MinerEpochResult(miner_id, None, copy_reason)

        return PreparedMinerSubmission(
            miner_id=miner_id,
            pointer=pointer,
            adapter_files=adapter.files,
            adapter_hash=self._adapter_hash(adapter.files),
        )

    def _math_rollouts_for_prepared_miner(
        self,
        prepared: PreparedMinerSubmission,
        batch: list[ProblemInstance],
        original_rollouts: list[OriginalRollout],
        *,
        precomputed_math_completions: list[tuple[str, int]] | None = None,
        precomputed_math_scores: list[float] | None = None,
        sample_weights: dict[tuple[str, str], float] | None = None,
    ) -> tuple[list[RolloutResult], str | None]:
        miner_id = prepared.miner_id
        if len(original_rollouts) != len(batch):
            return [], (
                f"baseline_mismatch: {len(original_rollouts)} rollouts "
                f"for {len(batch)} problems"
            )
        if not batch:
            return [], None

        prompts = [instance.prompt for instance in batch]
        if precomputed_math_completions is None:
            budgets = [
                self._generation_budget(original.completion_len)
                for original in original_rollouts
            ]
            self._log(f"miner {miner_id} math: generating {len(prompts)} completion(s)")
            try:
                completions = self._generate_miner_with_budgets(
                    miner_id,
                    prepared.adapter_files,
                    prompts,
                    budgets,
                )
            except Exception as exc:
                return [], f"inference_failed: {exc}"
        else:
            completions = precomputed_math_completions

        try:
            completions = self._validate_completion_count(miner_id, completions, len(batch))
        except Exception as exc:
            return [], f"inference_failed: {exc}"
        if precomputed_math_scores is not None and len(precomputed_math_scores) != len(batch):
            return [], (
                f"math_score_mismatch: {len(precomputed_math_scores)} scores "
                f"for {len(batch)} problems"
            )
        self._log(f"miner {miner_id} math: complete {len(completions)}/{len(prompts)}")
        if completions:
            self._log_sample_result(
                f"miner {miner_id} math",
                token_count=completions[0][1],
            )

        results: list[RolloutResult] = []
        for instance, original, (completion, token_count) in zip(
            batch, original_rollouts, completions
        ):
            track = get_track(instance.track)
            miner_verified = track.verify(instance.seed, completion)
            if precomputed_math_scores is None:
                rollout_score = relative_reasoning_reward(
                    original_verified=original.verified,
                    miner_verified=miner_verified,
                    original_completion_len=original.completion_len,
                    miner_completion_len=token_count,
                )
            else:
                rollout_score = precomputed_math_scores[len(results)]
            band = difficulty_band(instance.difficulty.percentile, self._config.n_difficulty_bands)
            results.append(
                RolloutResult(
                    track=instance.track,
                    seed=instance.seed,
                    band=band,
                    score=rollout_score,
                    sample_weight=self._sample_weight(
                        sample_weights, instance.track, instance.seed
                    ),
                    original_verified=original.verified,
                    miner_verified=miner_verified,
                    original_completion_len=original.completion_len,
                    miner_completion_len=token_count,
                )
            )
        return results, None

    def _long_context_rollouts_for_prepared_miner(
        self,
        prepared: PreparedMinerSubmission,
        batch: list[LongContextQAInstance],
        originals: list[LongContextAnswer] | None,
        *,
        precomputed_results: list[LongContextMinerResult] | None = None,
        sample_weights: dict[tuple[str, str], float] | None = None,
    ) -> tuple[list[RolloutResult], str | None]:
        miner_id = prepared.miner_id
        if not batch:
            return [], None
        if self._long_context is None:
            return [], "long_context_unavailable: evaluator missing"
        if originals is None:
            originals = self.score_original_long_context_batch(batch)
        if len(originals) != len(batch):
            return [], (
                f"long_context_baseline_mismatch: {len(originals)} "
                f"answers for {len(batch)} questions"
            )
        try:
            self._log(
                f"miner {miner_id} long-context QA: scoring "
                f"{len(batch)} sample(s)"
            )
            if precomputed_results is not None:
                long_context_results = precomputed_results
            else:
                long_context_results = self._long_context.score_miner_batch(
                    miner_id=miner_id,
                    adapter_files=prepared.adapter_files,
                    instances=batch,
                    originals=originals,
                )
            if len(long_context_results) != len(batch):
                return [], (
                    f"long_context_mismatch: {len(long_context_results)} results "
                    f"for {len(batch)} questions"
                )
            self._log(
                f"miner {miner_id} long-context QA: complete "
                f"{len(long_context_results)}/{len(batch)}"
            )
            if long_context_results:
                first_result = long_context_results[0]
                self._log_sample_result(
                    f"miner {miner_id} long-context QA",
                    token_count=first_result.miner.completion_len,
                    verified=first_result.miner.verified,
                    search_used=bool(first_result.search_query),
                )
            return [
                RolloutResult(
                    track="long_context_qa",
                    seed=instance.seed,
                    band="long_context_qa",
                    score=result.score,
                    sample_weight=self._sample_weight(
                        sample_weights, TASK_LONG_CONTEXT_QA, instance.seed
                    ),
                    original_verified=result.original.verified,
                    miner_verified=result.miner.verified,
                    original_completion_len=result.original.completion_len,
                    miner_completion_len=result.miner.completion_len,
                )
                for instance, result in zip(batch, long_context_results)
            ], None
        except Exception as exc:
            return [], f"long_context_failed: {exc}"

    def _score_prepared_miner(
        self,
        prepared: PreparedMinerSubmission,
        batch: list[ProblemInstance],
        original_rollouts: list[OriginalRollout],
        long_context_batch: list[LongContextQAInstance] | None = None,
        original_long_context: list[LongContextAnswer] | None = None,
        *,
        precomputed_math_completions: list[tuple[str, int]] | None = None,
        precomputed_math_scores: list[float] | None = None,
        precomputed_long_context_results: list[LongContextMinerResult] | None = None,
        sample_weights: dict[tuple[str, str], float] | None = None,
    ) -> MinerEpochResult:
        miner_id = prepared.miner_id
        math_rollouts, error = self._math_rollouts_for_prepared_miner(
            prepared,
            batch,
            original_rollouts,
            precomputed_math_completions=precomputed_math_completions,
            precomputed_math_scores=precomputed_math_scores,
            sample_weights=sample_weights,
        )
        if error is not None:
            return MinerEpochResult(miner_id, None, error)

        long_context_rollouts, error = self._long_context_rollouts_for_prepared_miner(
            prepared,
            long_context_batch or [],
            original_long_context,
            precomputed_results=precomputed_long_context_results,
            sample_weights=sample_weights,
        )
        if error is not None:
            return MinerEpochResult(miner_id, None, error)

        results = [*math_rollouts, *long_context_rollouts]
        score, task_scores = self._score_rollouts_by_task(
            results,
            min_coverage_per_band=self._config.min_coverage_per_band,
        )
        return self._scored_result(miner_id, score, results, task_scores)

    def _score_prepared_multiple_choice(
        self,
        prepared: PreparedMinerSubmission,
        batch: list[MultipleChoiceInstance],
        originals: list[MultipleChoiceAnswer],
        *,
        precomputed_results: list[MultipleChoiceMinerResult] | None = None,
        sample_weights: dict[tuple[str, str], float] | None = None,
    ) -> MinerEpochResult:
        miner_id = prepared.miner_id
        if not batch:
            return MinerEpochResult(
                miner_id,
                StratifiedScore(overall=0.0, per_band={}, coverage_ok=True, reason=None),
                None,
            )
        if self._multiple_choice is None:
            return MinerEpochResult(
                miner_id, None, "multiple_choice_unavailable: evaluator missing"
            )
        try:
            if precomputed_results is not None:
                scored = precomputed_results
            else:
                scored = self._multiple_choice.score_miner_batch(
                    miner_id=miner_id,
                    adapter_files=prepared.adapter_files,
                    instances=batch,
                    originals=originals,
                )
            if len(scored) != len(batch):
                return MinerEpochResult(
                    miner_id,
                    None,
                    f"multiple_choice_mismatch: {len(scored)} answers for {len(batch)} samples",
                )
            rollouts = [
                multiple_choice_rollout_result(
                    instance,
                    result,
                    sample_weight=self._sample_weight(
                        sample_weights, TASK_MULTIPLE_CHOICE, instance.seed
                    ),
                )
                for instance, result in zip(batch, scored)
            ]
            score, task_scores = self._score_rollouts_by_task(
                rollouts,
                min_coverage_per_band=1,
            )
            return self._scored_result(miner_id, score, rollouts, task_scores)
        except Exception as exc:
            return MinerEpochResult(miner_id, None, f"multiple_choice_failed: {exc}")

    def _combine_scored_results(
        self, miner_id: str, results: list[MinerEpochResult]
    ) -> MinerEpochResult:
        failures = [result for result in results if result.score is None]
        if failures:
            return failures[0]
        scores = [result.score for result in results if result.score is not None]
        if not scores:
            return MinerEpochResult(
                miner_id,
                StratifiedScore(overall=0.0, per_band={}, coverage_ok=True, reason=None),
                None,
            )
        per_band: dict[str, float] = {}
        for score in scores:
            per_band.update(score.per_band)
        coverage_ok = all(score.coverage_ok for score in scores)
        reason = "; ".join(score.reason for score in scores if score.reason) or None
        telemetry_results = [result for result in results if result.telemetry_count > 0]

        def mean_task_values(field: str) -> dict[str, float]:
            values: dict[str, list[float]] = defaultdict(list)
            for result in telemetry_results:
                for task, value in getattr(result, field).items():
                    values[task].append(float(value))
            return {
                task: sum(task_values) / len(task_values)
                for task, task_values in values.items()
                if task_values
            }

        task_scores = mean_task_values("task_scores")
        for task, score in sorted(task_scores.items()):
            per_band[f"type/{task}"] = score
        weights = self._score_type_weights()
        weighted_overall = self._weighted_task_mean(task_scores, weights)
        overall = (
            weighted_overall
            if weighted_overall is not None
            else sum(score.overall for score in scores) / len(scores)
        )
        task_completion_len = mean_task_values("task_completion_len")
        task_correctness_score = mean_task_values("task_correctness_score")
        task_original_score = mean_task_values("task_original_score")
        task_original_correctness_score = mean_task_values(
            "task_original_correctness_score"
        )
        task_original_completion_len = mean_task_values(
            "task_original_completion_len"
        )

        return MinerEpochResult(
            miner_id,
            StratifiedScore(
                overall=overall,
                per_band=per_band,
                coverage_ok=coverage_ok,
                reason=reason,
            ),
            None,
            completion_len=self._weighted_task_mean(task_completion_len, weights),
            correctness_score=self._weighted_task_mean(task_correctness_score, weights),
            original_score=self._weighted_task_mean(task_original_score, weights),
            original_correctness_score=self._weighted_task_mean(
                task_original_correctness_score, weights
            ),
            original_completion_len=self._weighted_task_mean(
                task_original_completion_len, weights
            ),
            telemetry_count=sum(
                result.telemetry_count for result in telemetry_results
            ),
            task_scores=task_scores,
            task_completion_len=task_completion_len,
            task_correctness_score=task_correctness_score,
            task_original_score=task_original_score,
            task_original_correctness_score=task_original_correctness_score,
            task_original_completion_len=task_original_completion_len,
        )

    def _full_eval_key(self, n_math: int, n_long_context_qa: int) -> str:
        return (
            f"base={self._config.base_model_repo}@{self._config.base_model_revision}|"
            f"math={n_math}|long_context_qa={n_long_context_qa}"
        )

    def _history_score(self, prepared: PreparedMinerSubmission, eval_key: str) -> float | None:
        cached = self._eval_cache.get(prepared.adapter_hash, eval_key)
        return cached.score if cached is not None else None

    def _history_score_from_snapshot(
        self,
        prepared: PreparedMinerSubmission,
        eval_key: str,
        history_snapshot: dict[str, float | None] | None,
    ) -> float | None:
        if history_snapshot is None:
            return self._history_score(prepared, eval_key)
        return history_snapshot.get(prepared.adapter_hash)

    def _rank_score(
        self,
        prepared: PreparedMinerSubmission,
        qualification: MinerEpochResult,
        eval_key: str,
        history_snapshot: dict[str, float | None] | None = None,
    ) -> float:
        current = qualification.score.overall if qualification.score is not None else 0.0
        history = self._history_score_from_snapshot(
            prepared, eval_key, history_snapshot
        )
        if history is None:
            return current
        history_weight = min(1.0, max(0.0, self._config.full_eval_history_weight))
        return (1.0 - history_weight) * current + history_weight * history

    def _apply_full_eval_ema(
        self,
        prepared: PreparedMinerSubmission,
        full_result: MinerEpochResult,
        eval_key: str,
        history_snapshot: dict[str, float | None] | None = None,
    ) -> MinerEpochResult:
        if full_result.score is None or not full_result.score.coverage_ok:
            return full_result
        history = self._history_score_from_snapshot(
            prepared, eval_key, history_snapshot
        )
        alpha = min(1.0, max(0.0, self._config.full_eval_ema_alpha))
        raw_score = full_result.score.overall
        smoothed = raw_score if history is None else alpha * raw_score + (1.0 - alpha) * history
        per_band = dict(full_result.score.per_band)
        per_band["full_eval_raw"] = raw_score
        if history is not None:
            per_band["full_eval_history"] = history
        return MinerEpochResult(
            full_result.miner_id,
            StratifiedScore(
                overall=smoothed,
                per_band=per_band,
                coverage_ok=full_result.score.coverage_ok,
                reason=full_result.score.reason,
            ),
            full_result.rejected_reason,
            completion_len=full_result.completion_len,
            correctness_score=full_result.correctness_score,
            original_score=full_result.original_score,
            original_correctness_score=full_result.original_correctness_score,
            original_completion_len=full_result.original_completion_len,
            telemetry_count=full_result.telemetry_count,
            task_scores=full_result.task_scores,
            task_completion_len=full_result.task_completion_len,
            task_correctness_score=full_result.task_correctness_score,
            task_original_score=full_result.task_original_score,
            task_original_correctness_score=(
                full_result.task_original_correctness_score
            ),
            task_original_completion_len=full_result.task_original_completion_len,
        )

    def _qualification_only_score(
        self,
        prepared: PreparedMinerSubmission,
        qualification: MinerEpochResult,
        eval_key: str,
        qualification_items: int,
        full_items: int,
        history_snapshot: dict[str, float | None] | None = None,
    ) -> MinerEpochResult:
        if qualification.score is None:
            return qualification
        rank_score = self._rank_score(
            prepared, qualification, eval_key, history_snapshot
        )
        confidence = 0.0 if full_items <= 0 else min(1.0, qualification_items / full_items)
        overall = rank_score * confidence
        history = self._history_score_from_snapshot(
            prepared, eval_key, history_snapshot
        )
        per_band = {"qualification": qualification.score.overall, "confidence": confidence}
        per_band.update(
            {
                f"type/{task}": score
                for task, score in sorted(qualification.task_scores.items())
            }
        )
        if history is not None:
            per_band["history"] = history
        return MinerEpochResult(
            prepared.miner_id,
            StratifiedScore(
                overall=overall,
                per_band=per_band,
                coverage_ok=qualification.score.coverage_ok,
                reason="qualification_only",
            ),
            None,
            completion_len=qualification.completion_len,
            correctness_score=qualification.correctness_score,
            original_score=qualification.original_score,
            original_correctness_score=qualification.original_correctness_score,
            original_completion_len=qualification.original_completion_len,
            telemetry_count=qualification.telemetry_count,
            task_scores=qualification.task_scores,
            task_completion_len=qualification.task_completion_len,
            task_correctness_score=qualification.task_correctness_score,
            task_original_score=qualification.task_original_score,
            task_original_correctness_score=(
                qualification.task_original_correctness_score
            ),
            task_original_completion_len=qualification.task_original_completion_len,
        )

    def _select_full_eval_miners(
        self,
        prepared: dict[str, PreparedMinerSubmission],
        qualification_results: dict[str, MinerEpochResult],
        eval_key: str,
        history_snapshot: dict[str, float | None] | None = None,
    ) -> set[str]:
        ranked: list[tuple[float, str]] = []
        for miner_id, prepared_submission in prepared.items():
            qualification = qualification_results.get(miner_id)
            if qualification is None or qualification.score is None or not qualification.score.coverage_ok:
                continue
            ranked.append((
                self._rank_score(
                    prepared_submission, qualification, eval_key, history_snapshot
                ),
                miner_id,
            ))
        ranked.sort(reverse=True)

        top_k = max(0, self._config.full_eval_top_k)
        selected = {miner_id for _score, miner_id in ranked[:top_k]}
        # Carry forward the last few rounds' winners so a champion stays
        # eligible for full evaluation even if it resubmits a new adapter and
        # has a rough qualification round.
        selected |= self._round_state.recent_champions() & set(prepared)
        return selected

    def _winner_take_all_scores(
        self,
        results: dict[str, MinerEpochResult],
        *,
        candidates: set[str] | None = None,
    ) -> tuple[dict[str, float], str | None]:
        eligible = {
            miner_id: r.score.overall
            for miner_id, r in results.items()
            if r.score is not None and r.score.coverage_ok
        }
        pool = eligible if candidates is None else {
            miner_id: score for miner_id, score in eligible.items() if miner_id in candidates
        }
        winner_id = None
        if pool:
            winner_id = max(pool.items(), key=lambda kv: (kv[1], kv[0]))[0]
        scores = {miner_id: (1.0 if miner_id == winner_id else 0.0) for miner_id in eligible}
        return scores, winner_id

    def _set_weights_from_results(
        self,
        results: dict[str, MinerEpochResult],
        *,
        full_eval_miners: set[str] | None = None,
        weight_candidate_miners: set[str] | None = None,
    ) -> str | None:
        candidates = weight_candidate_miners if weight_candidate_miners is not None else full_eval_miners
        scores, winner_id = self._winner_take_all_scores(results, candidates=candidates)
        self._weights.set_weights(scores)
        return winner_id

    def _run_full_epoch(self, pointers: dict[str, MinerSubmissionPointer], epoch_batch: EpochBatch) -> dict[str, MinerEpochResult]:
        self._log(
            "evaluation: scoring baseline "
            f"({len(epoch_batch.math)} math, {len(epoch_batch.long_context_qa)} long-context QA)"
        )
        original_rollouts = self.score_original_batch(epoch_batch.math)
        original_long_context = self.score_original_long_context_batch(
            epoch_batch.long_context_qa
        )

        prepared: dict[str, PreparedMinerSubmission] = {}
        results: dict[str, MinerEpochResult] = {}
        with self._progress(
            len(pointers), "submission checks", unit="miner"
        ) as progress:
            for miner_id, pointer in pointers.items():
                prepared_or_rejected = self.prepare_miner(miner_id, pointer)
                if isinstance(prepared_or_rejected, MinerEpochResult):
                    results[miner_id] = prepared_or_rejected
                else:
                    prepared[miner_id] = prepared_or_rejected
                progress.update(1)
                progress.set_postfix(
                    accepted=len(prepared), rejected=len(results), refresh=False
                )

        math_completions, math_errors = self._generate_miners_math_completions_batch(
            [(miner_id, submission.adapter_files) for miner_id, submission in prepared.items()],
            epoch_batch.math,
            original_rollouts,
        )
        math_scores = self._score_miners_math_completions_batch(
            epoch_batch.math,
            original_rollouts,
            math_completions,
        )
        long_context_results, long_context_errors = self._score_miners_long_context_batch(
            list(prepared.items()),
            epoch_batch.long_context_qa,
            original_long_context,
        )
        sample_weights = {
            **self._math_sample_weights(epoch_batch.math, math_completions),
            **self._long_context_sample_weights(
                epoch_batch.long_context_qa, long_context_results
            ),
        }

        total = len(prepared)
        for idx, (miner_id, prepared_submission) in enumerate(prepared.items(), start=1):
            start = time.monotonic()
            self._log(f"evaluation: scoring miner {idx}/{total} {miner_id}")
            if miner_id in math_errors:
                result = MinerEpochResult(
                    miner_id, None, f"inference_failed: {math_errors[miner_id]}"
                )
            elif miner_id in long_context_errors:
                result = MinerEpochResult(
                    miner_id, None, f"long_context_failed: {long_context_errors[miner_id]}"
                )
            else:
                result = self._score_prepared_miner(
                    prepared_submission,
                    epoch_batch.math,
                    original_rollouts,
                    epoch_batch.long_context_qa,
                    original_long_context,
                    precomputed_math_completions=math_completions.get(miner_id, []),
                    precomputed_math_scores=math_scores.get(miner_id),
                    precomputed_long_context_results=long_context_results.get(miner_id),
                    sample_weights=sample_weights,
                )
            results[miner_id] = result
            self._log(
                f"evaluation: miner {idx}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; {self._score_summary(result)}"
            )
        self._log_results_table(results)
        self._log("evaluation: setting weights")
        self._set_weights_from_results(results)
        return results

    def _prepare_test_submissions(
        self,
        pointers: dict[str, MinerSubmissionPointer],
    ) -> tuple[dict[str, PreparedMinerSubmission], dict[str, MinerEpochResult]]:
        prepared: dict[str, PreparedMinerSubmission] = {}
        results: dict[str, MinerEpochResult] = {}
        rejected_messages: list[str] = []
        with self._progress(
            len(pointers), "submission checks", unit="miner"
        ) as progress:
            for miner_id, pointer in pointers.items():
                prepared_or_rejected = self.prepare_miner(miner_id, pointer)
                if isinstance(prepared_or_rejected, MinerEpochResult):
                    results[miner_id] = prepared_or_rejected
                    rejected_messages.append(
                        f"submission checks: {miner_id} rejected: "
                        f"{prepared_or_rejected.rejected_reason}"
                    )
                else:
                    prepared[miner_id] = prepared_or_rejected
                progress.update(1)
                progress.set_postfix(
                    accepted=len(prepared), rejected=len(results), refresh=False
                )

        for message in rejected_messages:
            self._log(message)
        print(
            f"[thinker-validator] submission checks: {len(prepared)} accepted, "
            f"{len(results)} rejected",
            flush=True,
        )
        return prepared, results

    def _run_test_math_epoch(
        self,
        prepared: dict[str, PreparedMinerSubmission],
        *,
        n_problems: int,
        epoch: int,
        common_seed: str | None,
    ) -> dict[str, MinerEpochResult]:
        self._log(
            f"test mode math: scoring baseline ({n_problems} math sample(s))"
        )
        batch = self.build_math_batch(
            n_problems,
            epoch=epoch,
            common_seed=common_seed,
            seed_namespace="test_mode",
        )
        original_rollouts = self.score_original_batch(batch)
        math_completions, math_errors = self._generate_miners_math_completions_batch(
            [
                (miner_id, submission.adapter_files)
                for miner_id, submission in prepared.items()
            ],
            batch,
            original_rollouts,
        )
        math_scores = self._score_miners_math_completions_batch(
            batch,
            original_rollouts,
            math_completions,
        )
        sample_weights = self._math_sample_weights(batch, math_completions)
        self._log_test_math_samples(
            batch,
            original_rollouts,
            math_completions,
            math_scores,
            math_errors,
            list(prepared),
        )

        results: dict[str, MinerEpochResult] = {}
        total = len(prepared)
        for idx, (miner_id, prepared_submission) in enumerate(prepared.items(), start=1):
            start = time.monotonic()
            self._log(f"test mode math: scoring miner {idx}/{total} {miner_id}")
            if miner_id in math_errors:
                result = MinerEpochResult(
                    miner_id, None, f"inference_failed: {math_errors[miner_id]}"
                )
            else:
                result = self._score_prepared_miner(
                    prepared_submission,
                    batch,
                    original_rollouts,
                    [],
                    [],
                    precomputed_math_completions=math_completions.get(miner_id, []),
                    precomputed_math_scores=math_scores.get(miner_id),
                    precomputed_long_context_results=[],
                    sample_weights=sample_weights,
                )
            results[miner_id] = result
            self._log(
                f"test mode math: miner {idx}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; {self._score_summary(result)}"
            )
        return results

    def _run_test_long_context_epoch(
        self,
        prepared: dict[str, PreparedMinerSubmission],
        *,
        n_questions: int,
        epoch: int,
        common_seed: str | None,
    ) -> dict[str, MinerEpochResult]:
        self._log(
            "test mode long_qa: scoring baseline "
            f"({n_questions} long-context QA sample(s))"
        )
        batch = self.build_long_context_batch(
            n_questions,
            epoch=epoch,
            common_seed=common_seed,
            seed_namespace="test_mode",
        )
        self._log_test_long_context_generated_samples(batch)
        original_long_context = self.score_original_long_context_batch(batch)
        long_context_results, long_context_errors = (
            self._score_miners_long_context_batch(
                list(prepared.items()),
                batch,
                original_long_context,
            )
        )
        sample_weights = self._long_context_sample_weights(
            batch, long_context_results
        )
        self._log_test_long_context_samples(
            batch,
            original_long_context,
            long_context_results,
            long_context_errors,
        )

        results: dict[str, MinerEpochResult] = {}
        total = len(prepared)
        for idx, (miner_id, prepared_submission) in enumerate(prepared.items(), start=1):
            start = time.monotonic()
            self._log(f"test mode long_qa: scoring miner {idx}/{total} {miner_id}")
            if miner_id in long_context_errors:
                result = MinerEpochResult(
                    miner_id,
                    None,
                    f"long_context_failed: {long_context_errors[miner_id]}",
                )
            else:
                result = self._score_prepared_miner(
                    prepared_submission,
                    [],
                    [],
                    batch,
                    original_long_context,
                    precomputed_math_completions=[],
                    precomputed_math_scores=[],
                    precomputed_long_context_results=long_context_results.get(miner_id),
                    sample_weights=sample_weights,
                )
            results[miner_id] = result
            self._log(
                f"test mode long_qa: miner {idx}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; {self._score_summary(result)}"
            )
        return results

    def _run_test_science_epoch(
        self,
        prepared: dict[str, PreparedMinerSubmission],
        *,
        n_samples: int,
        epoch: int,
        common_seed: str | None,
    ) -> dict[str, MinerEpochResult]:
        thinking_count = max(0, n_samples)
        self._log(
            "test mode science: scoring baseline "
            f"({n_samples} multiple-choice sample(s), "
            f"{thinking_count} thinking, 0 no-thinking)"
        )
        batch = self.build_multiple_choice_batch(
            n_samples,
            thinking_count,
            epoch=epoch,
            common_seed=common_seed,
            seed_namespace="test_mode",
        )
        originals = self.score_original_multiple_choice_batch(batch)
        multiple_choice_results, multiple_choice_errors = (
            self._score_miners_multiple_choice_batch(
                list(prepared.items()),
                batch,
                originals,
            )
        )
        sample_weights = self._multiple_choice_sample_weights(
            batch, multiple_choice_results
        )
        self._log_test_science_samples(
            batch,
            originals,
            multiple_choice_results,
            multiple_choice_errors,
            list(prepared),
        )

        results: dict[str, MinerEpochResult] = {}
        total = len(prepared)
        for idx, (miner_id, prepared_submission) in enumerate(prepared.items(), start=1):
            start = time.monotonic()
            self._log(f"test mode science: scoring miner {idx}/{total} {miner_id}")
            if miner_id in multiple_choice_errors:
                result = MinerEpochResult(
                    miner_id,
                    None,
                    f"multiple_choice_failed: {multiple_choice_errors[miner_id]}",
                )
            else:
                result = self._score_prepared_multiple_choice(
                    prepared_submission,
                    batch,
                    originals,
                    precomputed_results=multiple_choice_results.get(miner_id),
                    sample_weights=sample_weights,
                )
            results[miner_id] = result
            self._log(
                f"test mode science: miner {idx}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; {self._score_summary(result)}"
            )
        return results

    def _run_test_epoch(
        self,
        pointers: dict[str, MinerSubmissionPointer],
        *,
        mode: str,
        n_problems: int | None,
        n_long_context_qa: int | None,
        n_multiple_choice: int | None,
        epoch: int,
        common_seed: str | None,
    ) -> dict[str, MinerEpochResult]:
        self._log(
            f"test mode {mode}: task-only full evaluation; "
            "skipping qualification, weights, cache, and round-state updates"
        )
        prepared, rejected_results = self._prepare_test_submissions(pointers)
        if not prepared:
            self._log_results_table(rejected_results)
            return rejected_results

        if mode == "math":
            scored_results = self._run_test_math_epoch(
                prepared,
                n_problems=(
                    n_problems
                    if n_problems is not None
                    else self._config.n_problems_per_epoch
                ),
                epoch=epoch,
                common_seed=common_seed,
            )
        elif mode == "long_qa":
            scored_results = self._run_test_long_context_epoch(
                prepared,
                n_questions=(
                    n_long_context_qa
                    if n_long_context_qa is not None
                    else self._config.n_long_context_qa_per_epoch
                ),
                epoch=epoch,
                common_seed=common_seed,
            )
        elif mode == "science":
            scored_results = self._run_test_science_epoch(
                prepared,
                n_samples=(
                    n_multiple_choice
                    if n_multiple_choice is not None
                    else self._config.qualification_multiple_choice_per_epoch
                ),
                epoch=epoch,
                common_seed=common_seed,
            )
        else:
            raise ValueError(f"unknown test mode: {mode}")

        results = {**rejected_results, **scored_results}
        self._log_results_table(results, full_eval_miners=set(scored_results))
        self._log(
            f"test mode {mode}: complete; scores were not written to chain"
        )
        return results

    def _prepare_staged_submissions(
        self,
        pointers: dict[str, MinerSubmissionPointer],
        epoch: int,
    ) -> tuple[
        dict[str, PreparedMinerSubmission],
        dict[str, MinerEpochResult],
        dict[str, str],
    ]:
        prepared: dict[str, PreparedMinerSubmission] = {}
        results: dict[str, MinerEpochResult] = {}
        rejected_messages: list[str] = []
        with self._progress(
            len(pointers), "submission checks", unit="miner"
        ) as progress:
            for miner_id, pointer in pointers.items():
                prepared_or_rejected = self.prepare_miner(miner_id, pointer)
                if isinstance(prepared_or_rejected, MinerEpochResult):
                    results[miner_id] = prepared_or_rejected
                    rejected_messages.append(
                        f"submission checks: {miner_id} rejected: "
                        f"{prepared_or_rejected.rejected_reason}"
                    )
                else:
                    prepared[miner_id] = prepared_or_rejected
                progress.update(1)
                progress.set_postfix(
                    accepted=len(prepared), rejected=len(results), refresh=False
                )

        for message in rejected_messages:
            self._log(message)
        print(
            f"[thinker-validator] submission checks: {len(prepared)} accepted, "
            f"{len(results)} rejected",
            flush=True,
        )

        rejected_progress = {miner_id: "rejected" for miner_id in results}
        self._report_progress(
            epoch,
            "qualification",
            "preparing",
            {
                **{miner_id: "pending" for miner_id in prepared},
                **rejected_progress,
            },
        )
        return prepared, results, rejected_progress

    def _qualification_counts(
        self,
        full_math: int,
        full_long_context: int,
    ) -> tuple[int, int, int, int]:
        math = min(
            max(0, self._config.qualification_math_per_epoch),
            full_math,
        )
        long_context = min(
            max(0, self._config.qualification_long_context_qa_per_epoch),
            full_long_context,
        )
        multiple_choice = max(
            0, self._config.qualification_multiple_choice_per_epoch
        )
        multiple_choice_thinking = multiple_choice
        return math, long_context, multiple_choice, multiple_choice_thinking

    def _build_qualification_baselines(
        self,
        *,
        math_count: int,
        long_context_count: int,
        multiple_choice_count: int,
        multiple_choice_thinking_count: int,
        epoch: int,
        common_seed: str | None,
    ) -> tuple[
        EpochBatch,
        list[OriginalRollout],
        list[LongContextAnswer],
        list[MultipleChoiceInstance],
        list[MultipleChoiceAnswer],
    ]:
        self._report_progress(
            epoch,
            "qualification",
            "evaluating",
            baseline_status="evaluating",
        )
        qualification_batch = EpochBatch(math=[], long_context_qa=[])
        original_math: list[OriginalRollout] = []
        original_long_context: list[LongContextAnswer] = []
        try:
            if math_count or long_context_count:
                self._log(
                    "qualification: building math/long-context evaluation batch"
                )
                qualification_batch = self.build_epoch_batch(
                    math_count,
                    long_context_count,
                    epoch=epoch,
                    common_seed=common_seed,
                    seed_namespace="qualification",
                )
                self._log(
                    "qualification: scoring baseline "
                    f"({len(qualification_batch.math)} math, "
                    f"{len(qualification_batch.long_context_qa)} long-context QA)"
                )
                original_math = self.score_original_batch(
                    qualification_batch.math
                )
                original_long_context = (
                    self.score_original_long_context_batch(
                        qualification_batch.long_context_qa
                    )
                )
            multiple_choice_batch = self.build_multiple_choice_batch(
                multiple_choice_count,
                multiple_choice_thinking_count,
                epoch=epoch,
                common_seed=common_seed,
                seed_namespace="qualification",
            )
            original_multiple_choice = (
                self.score_original_multiple_choice_batch(
                    multiple_choice_batch
                )
            )
        except Exception:
            self._report_progress(
                epoch,
                "qualification",
                "failed",
                baseline_status="failed",
            )
            raise

        self._report_progress(
            epoch,
            "qualification",
            "evaluating",
            baseline_status="finished",
        )
        return (
            qualification_batch,
            original_math,
            original_long_context,
            multiple_choice_batch,
            original_multiple_choice,
        )

    def _runnable_qualification_miners(
        self,
        prepared: dict[str, PreparedMinerSubmission],
    ) -> tuple[list[tuple[str, PreparedMinerSubmission]], set[str]]:
        runnable: list[tuple[str, PreparedMinerSubmission]] = []
        skipped: set[str] = set()
        for miner_id, prepared_submission in prepared.items():
            if self._round_state.should_skip_qualification(
                miner_id,
                prepared_submission.adapter_hash,
                skip_after_rounds=self._config.full_eval_skip_after_rounds,
            ):
                skipped.add(miner_id)
                self._log(
                    f"qualification: skipping {miner_id} (no full-eval slot in "
                    f"{self._config.full_eval_skip_after_rounds}+ round(s) on the "
                    "same submission -- waiting for a new adapter)"
                )
            else:
                runnable.append((miner_id, prepared_submission))
        return runnable, skipped

    def _precompute_qualification_scores(
        self,
        *,
        runnable_miners: list[tuple[str, PreparedMinerSubmission]],
        qualification_batch: EpochBatch,
        original_math: list[OriginalRollout],
        original_long_context: list[LongContextAnswer],
        multiple_choice_batch: list[MultipleChoiceInstance],
        original_multiple_choice: list[MultipleChoiceAnswer],
    ) -> QualificationScoringCache:
        math_completions, math_errors = ({}, {})
        math_scores = {}
        if qualification_batch.math:
            math_completions, math_errors = (
                self._generate_miners_math_completions_batch(
                    [
                        (miner_id, submission.adapter_files)
                        for miner_id, submission in runnable_miners
                    ],
                    qualification_batch.math,
                    original_math,
                )
            )
            math_scores = self._score_miners_math_completions_batch(
                qualification_batch.math,
                original_math,
                math_completions,
            )

        long_context_results, long_context_errors = ({}, {})
        if qualification_batch.long_context_qa:
            long_context_results, long_context_errors = (
                self._score_miners_long_context_batch(
                    runnable_miners,
                    qualification_batch.long_context_qa,
                    original_long_context,
                )
            )

        multiple_choice_results, multiple_choice_errors = ({}, {})
        if multiple_choice_batch:
            multiple_choice_results, multiple_choice_errors = (
                self._score_miners_multiple_choice_batch(
                    runnable_miners,
                    multiple_choice_batch,
                    original_multiple_choice,
                )
            )

        sample_weights = {
            **self._math_sample_weights(qualification_batch.math, math_completions),
            **self._long_context_sample_weights(
                qualification_batch.long_context_qa, long_context_results
            ),
            **self._multiple_choice_sample_weights(
                multiple_choice_batch, multiple_choice_results
            ),
        }

        return QualificationScoringCache(
            math_completions=math_completions,
            math_scores=math_scores,
            math_errors=math_errors,
            long_context_results=long_context_results,
            long_context_errors=long_context_errors,
            multiple_choice_results=multiple_choice_results,
            multiple_choice_errors=multiple_choice_errors,
            sample_weights=sample_weights,
        )

    def _qualification_result_for_miner(
        self,
        *,
        miner_id: str,
        prepared_submission: PreparedMinerSubmission,
        qualification_batch: EpochBatch,
        original_math: list[OriginalRollout],
        original_long_context: list[LongContextAnswer],
        multiple_choice_batch: list[MultipleChoiceInstance],
        original_multiple_choice: list[MultipleChoiceAnswer],
        cache: QualificationScoringCache,
    ) -> MinerEpochResult:
        if miner_id in cache.math_errors:
            return MinerEpochResult(
                miner_id,
                None,
                f"inference_failed: {cache.math_errors[miner_id]}",
            )

        result_parts: list[MinerEpochResult] = []
        if qualification_batch.math or qualification_batch.long_context_qa:
            if miner_id in cache.long_context_errors:
                result_parts.append(
                    MinerEpochResult(
                        miner_id,
                        None,
                        "long_context_failed: "
                        f"{cache.long_context_errors[miner_id]}",
                    )
                )
            else:
                result_parts.append(
                    self._score_prepared_miner(
                        prepared_submission,
                        qualification_batch.math,
                        original_math,
                        qualification_batch.long_context_qa,
                        original_long_context,
                        precomputed_math_completions=(
                            cache.math_completions.get(miner_id, [])
                        ),
                        precomputed_math_scores=cache.math_scores.get(miner_id),
                        precomputed_long_context_results=(
                            cache.long_context_results.get(miner_id)
                        ),
                        sample_weights=cache.sample_weights,
                    )
                )
        if multiple_choice_batch:
            if miner_id in cache.multiple_choice_errors:
                result_parts.append(
                    MinerEpochResult(
                        miner_id,
                        None,
                        "multiple_choice_failed: "
                        f"{cache.multiple_choice_errors[miner_id]}",
                    )
                )
            else:
                result_parts.append(
                    self._score_prepared_multiple_choice(
                        prepared_submission,
                        multiple_choice_batch,
                        original_multiple_choice,
                        precomputed_results=(
                            cache.multiple_choice_results.get(miner_id)
                        ),
                        sample_weights=cache.sample_weights,
                    )
                )
        return self._combine_scored_results(miner_id, result_parts)

    def _score_qualification_miners(
        self,
        *,
        epoch: int,
        runnable_miners: list[tuple[str, PreparedMinerSubmission]],
        skipped_miner_ids: set[str],
        qualification_batch: EpochBatch,
        original_math: list[OriginalRollout],
        original_long_context: list[LongContextAnswer],
        multiple_choice_batch: list[MultipleChoiceInstance],
        original_multiple_choice: list[MultipleChoiceAnswer],
    ) -> dict[str, MinerEpochResult]:
        self._report_progress(
            epoch,
            "qualification",
            "evaluating",
            {
                **{
                    miner_id: "evaluating"
                    for miner_id, _prepared_submission in runnable_miners
                },
                **{miner_id: "skipped" for miner_id in skipped_miner_ids},
            },
        )
        cache = self._precompute_qualification_scores(
            runnable_miners=runnable_miners,
            qualification_batch=qualification_batch,
            original_math=original_math,
            original_long_context=original_long_context,
            multiple_choice_batch=multiple_choice_batch,
            original_multiple_choice=original_multiple_choice,
        )
        qualification_results: dict[str, MinerEpochResult] = {}
        total = len(runnable_miners)
        for index, (miner_id, prepared_submission) in enumerate(
            runnable_miners, start=1
        ):
            start = time.monotonic()
            self._log(
                f"qualification: scoring miner {index}/{total} {miner_id}"
            )
            result = self._qualification_result_for_miner(
                miner_id=miner_id,
                prepared_submission=prepared_submission,
                qualification_batch=qualification_batch,
                original_math=original_math,
                original_long_context=original_long_context,
                multiple_choice_batch=multiple_choice_batch,
                original_multiple_choice=original_multiple_choice,
                cache=cache,
            )
            qualification_results[miner_id] = result
            self._report_progress(
                epoch,
                "qualification",
                "evaluating",
                {
                    miner_id: (
                        "finished" if result.score is not None else "failed"
                    )
                },
                miner_scores=(
                    {miner_id: result.score.overall}
                    if result.score is not None
                    else None
                ),
            )
            self._log(
                f"qualification: miner {index}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; "
                f"{self._score_summary(result)}"
            )
        return qualification_results

    def _run_qualification_stage(
        self,
        *,
        prepared: dict[str, PreparedMinerSubmission],
        canonical_prepared: dict[str, PreparedMinerSubmission],
        duplicate_prepared_aliases: dict[str, str],
        full_math: int,
        full_long_context: int,
        eval_key: str,
        history_snapshot: dict[str, float | None],
        epoch: int,
        common_seed: str | None,
    ) -> tuple[
        dict[str, MinerEpochResult],
        set[str],
        set[str],
        set[str],
        int,
    ]:
        (
            math_count,
            long_context_count,
            multiple_choice_count,
            multiple_choice_thinking_count,
        ) = self._qualification_counts(full_math, full_long_context)
        qualification_items = (
            math_count + long_context_count + multiple_choice_count
        )
        print(
            f"[thinker-validator] qualification: {math_count} math sample(s), "
            f"{long_context_count} long-context QA sample(s), "
            f"{multiple_choice_count} multiple-choice sample(s) "
            f"({multiple_choice_thinking_count} full-thinking) for "
            f"{len(prepared)} miner(s)",
            flush=True,
        )

        if qualification_items == 0:
            self._report_progress(
                epoch,
                "qualification",
                "skipped",
                {miner_id: "skipped" for miner_id in prepared},
                baseline_status="skipped",
            )
            return (
                {},
                set(canonical_prepared),
                set(prepared),
                set(),
                qualification_items,
            )

        (
            qualification_batch,
            original_math,
            original_long_context,
            multiple_choice_batch,
            original_multiple_choice,
        ) = self._build_qualification_baselines(
            math_count=math_count,
            long_context_count=long_context_count,
            multiple_choice_count=multiple_choice_count,
            multiple_choice_thinking_count=multiple_choice_thinking_count,
            epoch=epoch,
            common_seed=common_seed,
        )
        runnable_miners, skipped_miner_ids = (
            self._runnable_qualification_miners(prepared)
        )
        qualification_results = self._score_qualification_miners(
            epoch=epoch,
            runnable_miners=runnable_miners,
            skipped_miner_ids=skipped_miner_ids,
            qualification_batch=qualification_batch,
            original_math=original_math,
            original_long_context=original_long_context,
            multiple_choice_batch=multiple_choice_batch,
            original_multiple_choice=original_multiple_choice,
        )
        weight_candidate_miners = self._select_full_eval_miners(
            canonical_prepared,
            qualification_results,
            eval_key,
            history_snapshot,
        )
        full_eval_miners = set(weight_candidate_miners)
        full_eval_miners.update(
            duplicate_miner
            for duplicate_miner, canonical_miner in (
                duplicate_prepared_aliases.items()
            )
            if canonical_miner in weight_candidate_miners
        )
        self._report_progress(epoch, "qualification", "completed")
        return (
            qualification_results,
            weight_candidate_miners,
            full_eval_miners,
            skipped_miner_ids,
            qualification_items,
        )

    def _build_full_evaluation_baseline(
        self,
        *,
        full_math: int,
        full_long_context: int,
        epoch: int,
        common_seed: str | None,
    ) -> tuple[
        EpochBatch,
        list[OriginalRollout],
        list[LongContextAnswer],
    ]:
        self._report_progress(
            epoch,
            "full_evaluation",
            "evaluating",
            baseline_status="evaluating",
        )
        try:
            self._log("full evaluation: building evaluation batch")
            full_batch = self.build_epoch_batch(
                full_math,
                full_long_context,
                epoch=epoch,
                common_seed=common_seed,
                seed_namespace="full_evaluation",
            )
            self._log(
                "full evaluation: scoring baseline "
                f"({len(full_batch.math)} math, "
                f"{len(full_batch.long_context_qa)} long-context QA)"
            )
            original_math = self.score_original_batch(full_batch.math)
            original_long_context = self.score_original_long_context_batch(
                full_batch.long_context_qa
            )
        except Exception:
            self._report_progress(
                epoch,
                "full_evaluation",
                "failed",
                baseline_status="failed",
            )
            raise

        self._report_progress(
            epoch,
            "full_evaluation",
            "evaluating",
            baseline_status="finished",
        )
        return full_batch, original_math, original_long_context

    def _full_evaluation_result_for_miner(
        self,
        *,
        miner_id: str,
        prepared: PreparedMinerSubmission,
        full_batch: EpochBatch,
        original_math: list[OriginalRollout],
        original_long_context: list[LongContextAnswer],
        math_completions: dict[str, list[tuple[str, int]]],
        math_scores: dict[str, list[float]],
        math_errors: dict[str, str],
        long_context_results: dict[str, list[LongContextMinerResult]],
        long_context_errors: dict[str, str],
        sample_weights: dict[tuple[str, str], float],
        eval_key: str,
        history_snapshot: dict[str, float | None],
    ) -> MinerEpochResult:
        if miner_id in math_errors:
            return MinerEpochResult(
                miner_id,
                None,
                f"inference_failed: {math_errors[miner_id]}",
            )
        if miner_id in long_context_errors:
            return MinerEpochResult(
                miner_id,
                None,
                f"long_context_failed: {long_context_errors[miner_id]}",
            )
        full_result = self._score_prepared_miner(
            prepared,
            full_batch.math,
            original_math,
            full_batch.long_context_qa,
            original_long_context,
            precomputed_math_completions=math_completions.get(miner_id, []),
            precomputed_math_scores=math_scores.get(miner_id),
            precomputed_long_context_results=long_context_results.get(miner_id),
            sample_weights=sample_weights,
        )
        return self._apply_full_eval_ema(
            prepared,
            full_result,
            eval_key,
            history_snapshot,
        )

    def _score_full_evaluation_miners(
        self,
        *,
        prepared: dict[str, PreparedMinerSubmission],
        results: dict[str, MinerEpochResult],
        full_eval_miners: set[str],
        full_batch: EpochBatch,
        original_math: list[OriginalRollout],
        original_long_context: list[LongContextAnswer],
        eval_key: str,
        history_snapshot: dict[str, float | None],
        epoch: int,
    ) -> None:
        ordered_miners = [
            miner_id for miner_id in prepared if miner_id in full_eval_miners
        ]
        self._report_progress(
            epoch,
            "full_evaluation",
            "evaluating",
            {miner_id: "evaluating" for miner_id in ordered_miners},
        )

        math_completions, math_errors = (
            self._generate_miners_math_completions_batch(
                [
                    (miner_id, prepared[miner_id].adapter_files)
                    for miner_id in ordered_miners
                ],
                full_batch.math,
                original_math,
            )
        )
        math_scores = self._score_miners_math_completions_batch(
            full_batch.math,
            original_math,
            math_completions,
        )
        long_context_results, long_context_errors = (
            self._score_miners_long_context_batch(
                [
                    (miner_id, prepared[miner_id])
                    for miner_id in ordered_miners
                ],
                full_batch.long_context_qa,
                original_long_context,
            )
        )
        sample_weights = {
            **self._math_sample_weights(full_batch.math, math_completions),
            **self._long_context_sample_weights(
                full_batch.long_context_qa, long_context_results
            ),
        }

        total = len(ordered_miners)
        for index, miner_id in enumerate(ordered_miners, start=1):
            start = time.monotonic()
            self._log(
                f"full evaluation: scoring miner {index}/{total} {miner_id}"
            )
            full_result = self._full_evaluation_result_for_miner(
                miner_id=miner_id,
                prepared=prepared[miner_id],
                full_batch=full_batch,
                original_math=original_math,
                original_long_context=original_long_context,
                math_completions=math_completions,
                math_scores=math_scores,
                math_errors=math_errors,
                long_context_results=long_context_results,
                long_context_errors=long_context_errors,
                sample_weights=sample_weights,
                eval_key=eval_key,
                history_snapshot=history_snapshot,
            )
            results[miner_id] = full_result
            self._report_progress(
                epoch,
                "full_evaluation",
                "evaluating",
                {
                    miner_id: (
                        "finished"
                        if full_result.score is not None
                        else "failed"
                    )
                },
                miner_scores=(
                    {miner_id: full_result.score.overall}
                    if full_result.score is not None
                    else None
                ),
            )
            self._log(
                f"full evaluation: miner {index}/{total} {miner_id} done in "
                f"{time.monotonic() - start:.1f}s; "
                f"{self._score_summary(full_result)}"
            )
            if (
                full_result.score is not None
                and full_result.score.coverage_ok
            ):
                self._eval_cache.put(
                    prepared[miner_id].adapter_hash,
                    eval_key,
                    full_result.score.overall,
                    epoch,
                    metadata={"miner_id": miner_id},
                )

    def _run_full_evaluation_stage(
        self,
        *,
        prepared: dict[str, PreparedMinerSubmission],
        results: dict[str, MinerEpochResult],
        rejected_progress: dict[str, str],
        full_eval_miners: set[str],
        skipped_miner_ids: set[str],
        full_math: int,
        full_long_context: int,
        eval_key: str,
        history_snapshot: dict[str, float | None],
        epoch: int,
        common_seed: str | None,
    ) -> None:
        print(
            f"[thinker-validator] full evaluation: selected "
            f"{len(full_eval_miners)}/{len(prepared)} miner(s) for "
            f"{full_math + full_long_context} sample(s) "
            f"({len(skipped_miner_ids)} skipped as stagnant)",
            flush=True,
        )
        stage_updates = {
            miner_id: (
                "pending"
                if miner_id in full_eval_miners
                else "skipped"
                if miner_id in skipped_miner_ids
                else "not_selected"
            )
            for miner_id in prepared
        }
        stage_updates.update(rejected_progress)
        self._report_progress(
            epoch,
            "full_evaluation",
            "preparing" if full_eval_miners else "skipped",
            stage_updates,
            baseline_status="pending" if full_eval_miners else "skipped",
        )
        if not full_eval_miners:
            return

        full_batch, original_math, original_long_context = (
            self._build_full_evaluation_baseline(
                full_math=full_math,
                full_long_context=full_long_context,
                epoch=epoch,
                common_seed=common_seed,
            )
        )
        self._score_full_evaluation_miners(
            prepared=prepared,
            results=results,
            full_eval_miners=full_eval_miners,
            full_batch=full_batch,
            original_math=original_math,
            original_long_context=original_long_context,
            eval_key=eval_key,
            history_snapshot=history_snapshot,
            epoch=epoch,
        )
        self._report_progress(epoch, "full_evaluation", "completed")

    def _finalize_staged_epoch(
        self,
        *,
        prepared: dict[str, PreparedMinerSubmission],
        results: dict[str, MinerEpochResult],
        qualification_results: dict[str, MinerEpochResult],
        skipped_miner_ids: set[str],
        full_eval_miners: set[str],
        weight_candidate_miners: set[str],
        eval_key: str,
        qualification_items: int,
        full_items: int,
        history_snapshot: dict[str, float | None],
        epoch: int,
    ) -> dict[str, MinerEpochResult]:
        for miner_id, prepared_submission in prepared.items():
            if miner_id in results:
                continue
            if miner_id in skipped_miner_ids:
                results[miner_id] = MinerEpochResult(
                    miner_id,
                    None,
                    "skipped_stagnant_submission",
                )
                continue
            results[miner_id] = self._qualification_only_score(
                prepared_submission,
                qualification_results[miner_id],
                eval_key,
                qualification_items,
                full_items,
                history_snapshot,
            )
            self._log(
                f"qualification-only: {miner_id}; "
                f"{self._score_summary(results[miner_id])}"
            )

        for miner_id, prepared_submission in prepared.items():
            if miner_id in skipped_miner_ids:
                continue
            self._round_state.record_round(
                miner_id,
                prepared_submission.adapter_hash,
                was_selected_for_full_eval=miner_id in full_eval_miners,
            )

        self._log_results_table(results, full_eval_miners=full_eval_miners)
        self._log("evaluation: setting weights")
        winner_id = self._set_weights_from_results(
            results,
            full_eval_miners=full_eval_miners,
            weight_candidate_miners=weight_candidate_miners,
        )
        if winner_id is not None and results[winner_id].score is not None:
            self._round_state.record_champion(
                epoch,
                winner_id,
                results[winner_id].score.overall,
            )
            self._log(
                f"winner-take-all: {winner_id} is the round {epoch} champion"
            )
        self._round_state.save()
        return results

    def _run_staged_epoch(
        self,
        pointers: dict[str, MinerSubmissionPointer],
        n_problems: int | None,
        n_long_context_qa: int | None,
        epoch: int,
        common_seed: str | None,
    ) -> dict[str, MinerEpochResult]:
        full_math = (
            n_problems
            if n_problems is not None
            else self._config.n_problems_per_epoch
        )
        full_long_context = (
            n_long_context_qa
            if n_long_context_qa is not None
            else self._config.n_long_context_qa_per_epoch
        )
        full_items = full_math + full_long_context
        eval_key = self._full_eval_key(full_math, full_long_context)

        prepared, results, rejected_progress = (
            self._prepare_staged_submissions(pointers, epoch)
        )
        if not prepared:
            self._report_progress(
                epoch,
                "qualification",
                "completed",
                rejected_progress,
                baseline_status="skipped",
            )
            self._report_progress(
                epoch,
                "full_evaluation",
                "skipped",
                rejected_progress,
                baseline_status="skipped",
            )
            self._set_weights_from_results(results)
            return results

        history_snapshot = {
            prepared_submission.adapter_hash: self._history_score(
                prepared_submission, eval_key
            )
            for prepared_submission in prepared.values()
        }
        canonical_items, duplicate_aliases = self._dedupe_prepared_miners(
            list(prepared.items()),
            label="staged evaluation",
        )
        canonical_prepared = dict(canonical_items)

        (
            qualification_results,
            weight_candidate_miners,
            full_eval_miners,
            skipped_miner_ids,
            qualification_items,
        ) = self._run_qualification_stage(
            prepared=prepared,
            canonical_prepared=canonical_prepared,
            duplicate_prepared_aliases=duplicate_aliases,
            full_math=full_math,
            full_long_context=full_long_context,
            eval_key=eval_key,
            history_snapshot=history_snapshot,
            epoch=epoch,
            common_seed=common_seed,
        )
        self._run_full_evaluation_stage(
            prepared=prepared,
            results=results,
            rejected_progress=rejected_progress,
            full_eval_miners=full_eval_miners,
            skipped_miner_ids=skipped_miner_ids,
            full_math=full_math,
            full_long_context=full_long_context,
            eval_key=eval_key,
            history_snapshot=history_snapshot,
            epoch=epoch,
            common_seed=common_seed,
        )
        return self._finalize_staged_epoch(
            prepared=prepared,
            results=results,
            qualification_results=qualification_results,
            skipped_miner_ids=skipped_miner_ids,
            full_eval_miners=full_eval_miners,
            weight_candidate_miners=weight_candidate_miners,
            eval_key=eval_key,
            qualification_items=qualification_items,
            full_items=full_items,
            history_snapshot=history_snapshot,
            epoch=epoch,
        )

    def run_epoch(
        self,
        pointers: dict[str, MinerSubmissionPointer],
        n_problems: int | None = None,
        batch: list[ProblemInstance] | None = None,
        n_long_context_qa: int | None = None,
        n_multiple_choice: int | None = None,
        epoch_batch: EpochBatch | None = None,
        epoch: int | None = None,
        common_seed: str | None = None,
        test_mode: str | None = None,
    ) -> dict[str, MinerEpochResult]:
        if test_mode is not None:
            return self._run_test_epoch(
                pointers,
                mode=test_mode,
                n_problems=n_problems,
                n_long_context_qa=n_long_context_qa,
                n_multiple_choice=n_multiple_choice,
                epoch=0 if epoch is None else epoch,
                common_seed=common_seed,
            )
        if epoch_batch is None:
            if batch is not None:
                epoch_batch = EpochBatch(math=batch, long_context_qa=[])
            else:
                return self._run_staged_epoch(
                    pointers,
                    n_problems=n_problems,
                    n_long_context_qa=n_long_context_qa,
                    epoch=0 if epoch is None else epoch,
                    common_seed=common_seed,
                )
        return self._run_full_epoch(pointers, epoch_batch)
