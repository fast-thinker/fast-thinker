from __future__ import annotations

import json
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from tqdm.auto import tqdm

from thinker.retrieval.bm25 import BM25RetrievalService, CorpusDocument, RetrievalHit, format_hits
from thinker.reward.relative import (
    peer_completion_efficiency_rewards,
    relative_reasoning_reward,
)

MINER_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question that may require evidence from a "
    "large external knowledge base you cannot see directly. Select the best-supported "
    "option, grounding your choice in retrieved evidence whenever you are not already "
    "certain.\n\n"
    "Tool available to you:\n"
    "- Retrieval search: if you need evidence, emit exactly one query wrapped as "
    "<search>your query</search> and then stop -- do not write anything else after "
    "</search>. The validator will run the search and insert the results as an "
    "<information>...</information> block immediately after your </search> tag, "
    "then let you continue from there.\n\n"
    "Rules:\n"
    "- You may think privately first using <think>...</think>.\n"
    "- You may issue at most one <search> call.\n"
    "- If you already know the answer with high confidence, skip search and answer "
    "directly.\n"
    "- Once you receive an <information> block, use only that evidence (plus "
    "anything you already reasoned about) to choose your final answer.\n"
    "- Always give your final answer as just the single letter of your chosen "
    "option in LaTeX boxed form (for example \\boxed{B}), with no extra explanation.\n"
    "- Never fabricate facts, sources, or search results."
)

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
LOOSE_SEARCH_RE = re.compile(r"<search>(.*?)<search>", re.DOTALL | re.IGNORECASE)
BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.DOTALL)
UNTRACKED_SPAN_RE = re.compile(r"<search>.*?</search>", re.DOTALL | re.IGNORECASE)
CHOICE_LETTER_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.:)]?\s")
CHOICE_LETTER_ONLY_RE = re.compile(r"^\s*\(?([A-Za-z])\)?\.?\s*$")


class LongContextInferenceBackend(Protocol):
    def generate_original_limited(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        ...

    def generate_original_greedy_limited(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool = True,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        ...

    def generate_original_samples(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int | None,
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
    ) -> list[list[tuple[str, int]]]:
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

    def count_tokens(self, text: str) -> int:
        ...

    def suppress_progress(self):
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


@dataclass(frozen=True)
class LongContextQAConfig:
    seed_context_topk: int = 5
    answer_context_topk: int = 100
    max_chars_per_doc: int | None = 2_000
    qa_generation_max_chars_per_doc: int | None = 800
    qa_generation_batch_size: int = 4
    qa_generation_max_new_tokens: int = 256
    qa_generation_max_attempts: int = 3
    judge_max_new_tokens: int = 64
    original_answer_max_new_tokens: int = 4096
    miner_search_extra_tokens: int = 512
    mc_distractor_samples: int = 10
    mc_distractor_max_new_tokens: int = 24
    mc_distractor_temperature: float = 0.8
    mc_distractor_top_p: float = 0.95


@dataclass(frozen=True)
class LongContextQAInstance:
    seed: str
    source_document: CorpusDocument
    seed_hits: tuple[RetrievalHit, ...]
    question: str
    gold_answer: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class LongContextAnswer:
    text: str
    completion_len: int
    verified: bool


@dataclass(frozen=True)
class LongContextMinerResult:
    score: float
    original: LongContextAnswer
    miner: LongContextAnswer
    search_query: str | None


@dataclass
class _BatchedMinerAnswerState:
    final_answers: dict[str, list[str | None]]
    completion_lens: dict[str, list[int]]
    search_queries: dict[str, list[str | None]]
    followup_requests: list[tuple[str, dict[str, bytes], str]]
    followup_budgets: list[int]
    followup_positions: list[tuple[str, int, int]]


def _first_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("model output did not contain a JSON object")


def parse_generated_qa(text: str) -> tuple[str, str]:
    data = _first_json_object(text)
    question = str(data.get("question") or "").strip()
    answer = str(data.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError("generated QA JSON must contain non-empty question and answer")
    return question, answer


def parse_judgement(text: str) -> bool:
    try:
        data = _first_json_object(text)
    except Exception:
        lowered = text.strip().lower()
        return lowered.startswith("yes") or lowered.startswith("true")
    value = data.get("equivalent")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return False


def parse_search_query(text: str) -> str | None:
    match = SEARCH_RE.search(text) or LOOSE_SEARCH_RE.search(text)
    if not match:
        return None
    query = match.group(1).strip()
    return query or None


def extract_final_answer(text: str) -> str:
    if "</information>" in text:
        text = text.rsplit("</information>", 1)[1]
    boxed_answers = BOXED_ANSWER_RE.findall(text)
    if boxed_answers:
        text = boxed_answers[-1]
    return text.strip()


def strip_untracked_spans(text: str) -> str:
    if "</information>" in text:
        text = text.rsplit("</information>", 1)[1]
    return UNTRACKED_SPAN_RE.sub("", text)


def inject_information_after_search(text: str, information_block: str) -> str:
    match = SEARCH_RE.search(text)
    if match:
        return f"{text[:match.end()]}{information_block}{text[match.end():]}"
    return f"{text.rstrip()}{information_block}"


def parse_mc_choice(text: str, options: tuple[str, ...]) -> str | None:
    """Deterministically resolve a model's multiple-choice response to one of
    `options`. Tries a leading letter (``B``, ``(B)``, ``B)``, ``B.``) first,
    then falls back to an exact (case-insensitive) match against the option
    text. Returns None if neither resolves -- callers should treat that as an
    incorrect choice, not a free pass.
    """
    candidate = text.strip()
    if not candidate:
        return None

    letter_match = CHOICE_LETTER_ONLY_RE.match(candidate) or CHOICE_LETTER_RE.match(candidate)
    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return options[index]

    normalized = candidate.lower()
    for option in options:
        if option.strip().lower() == normalized:
            return option
    return None


class LongContextQAEvaluator:
    def __init__(
        self,
        *,
        retriever: BM25RetrievalService,
        inference: LongContextInferenceBackend,
        config: LongContextQAConfig | None = None,
        show_progress: bool = False,
    ):
        self._retriever = retriever
        self._inference = inference
        self._config = config or LongContextQAConfig()
        self._show_progress = show_progress

    @contextmanager
    def _progress(self, total: int, description: str):
        with self._inference.suppress_progress(), tqdm(
            total=total,
            desc=f"[thinker-validator] {description}",
            unit="prompt",
            dynamic_ncols=True,
            disable=not self._show_progress,
        ) as progress:
            yield progress

    def _generate_original(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        enable_thinking: bool = False,
        greedy: bool = False,
        progress_label: str | None = "long-context original generation",
    ) -> list[tuple[str, int]]:
        generate = (
            self._inference.generate_original_greedy_limited
            if greedy
            else self._inference.generate_original_limited
        )

        if progress_label is None:
            return generate(
                prompts,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
        with self._progress(len(prompts), progress_label) as progress:
            completions = generate(
                prompts,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            progress.update(len(prompts))
            return completions

    def _generate_original_samples(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        progress_label: str,
    ) -> list[list[tuple[str, int]]]:
        total = len(prompts) * num_samples
        with self._progress(total, progress_label) as progress:
            completions = self._inference.generate_original_samples(
                prompts,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=False,
            )
            progress.update(total)
        if len(completions) != len(prompts):
            raise ValueError("original model must return one sample set per distractor prompt")
        if any(len(sample_set) != num_samples for sample_set in completions):
            raise ValueError(
                f"original model must return exactly {num_samples} distractors per prompt"
            )
        return completions

    def _generate_miner(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[tuple[str, int]]:
        kwargs: dict[str, object] = {"max_new_tokens": max_new_tokens}
        if stop is not None:
            kwargs["stop"] = stop
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        return self._inference.generate_limited(
            miner_id, adapter_files, prompts, **kwargs
        )

    def _generate_miner_with_budgets(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        prompts: list[str],
        budgets: list[int],
        *,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
        progress_label: str = "miner long-context generation",
    ) -> list[tuple[str, int]]:
        if len(prompts) != len(budgets):
            raise ValueError("prompts and budgets must have the same length")
        if not prompts:
            return []

        completions: list[tuple[str, int] | None] = [None] * len(prompts)
        grouped: dict[int, list[int]] = {}
        for index, budget in enumerate(budgets):
            grouped.setdefault(max(1, int(budget)), []).append(index)

        with self._progress(len(prompts), progress_label) as progress:
            for budget, indexes in grouped.items():
                chunk_prompts = [prompts[index] for index in indexes]
                chunk_completions = self._generate_miner(
                    miner_id,
                    adapter_files,
                    chunk_prompts,
                    max_new_tokens=budget,
                    stop=stop,
                    system_prompt=system_prompt,
                )
                if len(chunk_completions) != len(indexes):
                    raise ValueError("miner must return exactly one response per prompt")
                for index, completion in zip(indexes, chunk_completions):
                    completions[index] = completion
                progress.update(len(indexes))

        if any(completion is None for completion in completions):
            raise ValueError("miner budgeted generation left a missing completion")
        return [completion for completion in completions if completion is not None]

    def generate_instances(self, seeds: list[str]) -> list[LongContextQAInstance]:
        if not seeds:
            return []
        prepared: list[tuple[str, CorpusDocument, tuple[RetrievalHit, ...]]] = []
        prompts: list[str] = []
        for seed in seeds:
            source_document = self._retriever.random_document(seed)
            seed_query = source_document.contents or f"{source_document.title}\n{source_document.text}"
            seed_hits = tuple(
                self._retriever.search(seed_query, topk=self._config.seed_context_topk)
            )
            if not seed_hits:
                raise ValueError("could not retrieve seed context for long-context QA generation")
            prepared.append((seed, source_document, seed_hits))
            prompts.append(self._build_qa_generation_prompt(seed_hits))

        completions: list[tuple[str, int]] = []
        chunk_size = max(1, self._config.qa_generation_batch_size)
        with self._progress(len(prompts), "long-context QA generation") as progress:
            for start in range(0, len(prompts), chunk_size):
                chunk = prompts[start : start + chunk_size]
                completions.extend(
                    self._generate_original(
                        chunk,
                        max_new_tokens=self._config.qa_generation_max_new_tokens,
                        greedy=True,
                        progress_label=None,
                    )
                )
                progress.update(len(chunk))
        if len(completions) != len(prepared):
            raise ValueError("original model must return exactly one generated QA per seed")

        qa_pairs: list[tuple[str, CorpusDocument, tuple[RetrievalHit, ...], str, str]] = []
        for index, ((seed, source_document, seed_hits), (completion, _tokens)) in enumerate(
            zip(prepared, completions), start=1
        ):
            prompt = prompts[index - 1]
            question, gold_answer = self._parse_generated_qa_with_retries(
                seed=seed,
                source_document=source_document,
                seed_hits=seed_hits,
                prompt=prompt,
                completion=completion,
            )
            qa_pairs.append((seed, source_document, seed_hits, question, gold_answer))

        distractor_sets = self._generate_distractors_batch(
            [(question, gold_answer) for (_seed, _doc, _hits, question, gold_answer) in qa_pairs]
        )

        instances: list[LongContextQAInstance] = []
        for (seed, source_document, seed_hits, question, gold_answer), distractors in zip(
            qa_pairs, distractor_sets
        ):
            options = self._build_mc_options(seed, gold_answer, distractors)
            instances.append(
                LongContextQAInstance(
                    seed=seed,
                    source_document=source_document,
                    seed_hits=seed_hits,
                    question=question,
                    gold_answer=gold_answer,
                    options=options,
                )
            )
        return instances

    def _parse_generated_qa_with_retries(
        self,
        *,
        seed: str,
        source_document: CorpusDocument,
        seed_hits: tuple[RetrievalHit, ...],
        prompt: str,
        completion: str,
    ) -> tuple[str, str]:
        attempts = max(1, self._config.qa_generation_max_attempts)
        current_completion = completion
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return parse_generated_qa(current_completion)
            except Exception as exc:
                last_error = exc
                self._log_invalid_generated_qa(attempt, attempts, exc)
                if attempt >= attempts:
                    break
                repair_prompt = self._build_qa_repair_prompt(
                    prompt=prompt,
                    invalid_completion=current_completion,
                    error=exc,
                    seed_hits=seed_hits,
                )
                completions = self._generate_original(
                    [repair_prompt],
                    max_new_tokens=self._config.qa_generation_max_new_tokens,
                    greedy=True,
                    progress_label="long-context QA repair",
                )
                if len(completions) != 1:
                    last_error = ValueError(
                        "original model must return exactly one repaired QA"
                    )
                    break
                current_completion = completions[0][0]

        question, answer = self._fallback_generated_qa(source_document)
        print(
            "[thinker-validator] long-context QA generation: using fallback QA "
            f"after parse failure ({type(last_error).__name__})",
            flush=True,
        )
        return question, answer

    @staticmethod
    def _log_invalid_generated_qa(
        attempt: int,
        attempts: int,
        error: Exception,
    ) -> None:
        print(
            "[thinker-validator] long-context QA generation: invalid JSON "
            f"at attempt {attempt}/{attempts} ({type(error).__name__}); "
            "generated content redacted",
            flush=True,
        )

    @staticmethod
    def _fallback_generated_qa(source_document: CorpusDocument) -> tuple[str, str]:
        title = source_document.title.strip()
        if title:
            return (
                "What is the title of the retrieved source document?",
                title,
            )

        text = " ".join((source_document.text or source_document.contents).split())
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        answer = first_sentence[:200].strip() or "unknown"
        return (
            "What is the first sentence of the retrieved source document?",
            answer,
        )

    def _generate_distractors_batch(
        self, questions_and_gold: list[tuple[str, str]]
    ) -> list[tuple[str, ...]]:
        """Generate closed-book (no documents, no thinking) guesses from the
        original model to use as wrong multiple-choice options. This never
        touches miner-controlled text -- both the samples and the gold answer
        come from the validator's own model, so it isn't an exploitable judge
        call the way scoring a miner's free-text answer would be.
        """
        if not questions_and_gold:
            return []
        samples = max(1, self._config.mc_distractor_samples)
        prompts = [
            self._build_distractor_prompt(question)
            for question, _gold in questions_and_gold
        ]
        completion_sets = self._generate_original_samples(
            prompts,
            num_samples=samples,
            max_new_tokens=self._config.mc_distractor_max_new_tokens,
            temperature=self._config.mc_distractor_temperature,
            top_p=self._config.mc_distractor_top_p,
            progress_label="long-context distractor generation",
        )

        per_instance_unique: list[list[str]] = []
        judge_items: list[tuple[str, str, str]] = []
        judge_positions: list[tuple[int, str]] = []
        for idx, (question, gold) in enumerate(questions_and_gold):
            chunk = completion_sets[idx]
            seen: dict[str, str] = {}
            for text, _tokens in chunk:
                candidate = text.strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key not in seen:
                    seen[key] = candidate
            unique = list(seen.values())
            per_instance_unique.append(unique)
            for candidate in unique:
                judge_positions.append((idx, candidate))
                judge_items.append((question, gold, candidate))

        duplicates_of_gold = self._judge_answers_batch(judge_items) if judge_items else []
        keep: list[list[str]] = [[] for _ in questions_and_gold]
        for (idx, candidate), is_duplicate_of_gold in zip(judge_positions, duplicates_of_gold):
            if not is_duplicate_of_gold:
                keep[idx].append(candidate)
        return [tuple(options) for options in keep]

    def _build_mc_options(
        self, seed: str, gold_answer: str, distractors: tuple[str, ...]
    ) -> tuple[str, ...]:
        options = list(distractors) + [gold_answer]
        rng = random.Random(f"{seed}:mc_shuffle")
        rng.shuffle(options)
        return tuple(options)

    def score_miner_batch(
        self,
        *,
        miner_id: str,
        adapter_files: dict[str, bytes],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer] | None = None,
    ) -> list[LongContextMinerResult]:
        if not instances:
            return []
        if originals is None:
            originals = self.score_original_batch(instances)
        if len(originals) != len(instances):
            raise ValueError("originals must match instances")

        miner_answers = self._answer_with_miner_batch(
            miner_id, adapter_files, instances, originals=originals
        )
        results: list[LongContextMinerResult] = []
        for instance, original, (miner, search_query) in zip(instances, originals, miner_answers):
            score = self._mc_reward(original, miner.completion_len, miner.text, instance.gold_answer)
            results.append(
                LongContextMinerResult(
                    score=score,
                    original=original,
                    miner=miner,
                    search_query=search_query,
                )
            )
        return results

    def score_miners_batch(
        self,
        *,
        miners: list[tuple[str, dict[str, bytes]]],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
    ) -> dict[str, list[LongContextMinerResult]]:
        if len(originals) != len(instances):
            raise ValueError("originals must match instances")
        if not miners:
            return {}
        if not instances:
            return {miner_id: [] for miner_id, _adapter_files in miners}

        prompts = [
            self._build_miner_prompt(instance.question, instance.options)
            for instance in instances
        ]
        first_requests, first_budgets = self._batched_first_pass_requests(
            miners, prompts, originals
        )
        first_completions = self._generate_batched_first_pass(
            first_requests, first_budgets
        )
        state = self._collect_batched_first_pass(
            miners=miners,
            instances=instances,
            originals=originals,
            prompts=prompts,
            first_completions=first_completions,
        )
        self._complete_batched_followups(state)
        answers_by_miner = self._answers_by_miner(miners, instances, state)
        return self._score_batched_answers(
            miners=miners,
            instances=instances,
            originals=originals,
            answers_by_miner=answers_by_miner,
            search_queries=state.search_queries,
        )

    def _batched_first_pass_requests(
        self,
        miners: list[tuple[str, dict[str, bytes]]],
        prompts: list[str],
        originals: list[LongContextAnswer],
    ) -> tuple[list[tuple[str, dict[str, bytes], str]], list[int]]:
        budgets = [self._miner_budget(original) for original in originals]
        requests: list[tuple[str, dict[str, bytes], str]] = []
        request_budgets: list[int] = []
        for miner_id, adapter_files in miners:
            for prompt, budget in zip(prompts, budgets):
                requests.append((miner_id, adapter_files, prompt))
                request_budgets.append(budget)
        return requests, request_budgets

    def _generate_batched_first_pass(
        self,
        requests: list[tuple[str, dict[str, bytes], str]],
        budgets: list[int],
    ) -> list[tuple[str, int]]:
        with self._progress(
            len(requests), "miner long-context first pass"
        ) as progress:
            completions = self._inference.generate_for_miners_batch(
                requests,
                max_new_tokens_list=budgets,
                stop=["</search>"],
                system_prompt=MINER_SYSTEM_PROMPT,
            )
            progress.update(len(requests))
        if len(completions) != len(requests):
            raise ValueError("miner must return exactly one first response per question")
        return completions

    def _collect_batched_first_pass(
        self,
        *,
        miners: list[tuple[str, dict[str, bytes]]],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
        prompts: list[str],
        first_completions: list[tuple[str, int]],
    ) -> _BatchedMinerAnswerState:
        state = _BatchedMinerAnswerState(
            final_answers={
                miner_id: [None] * len(instances) for miner_id, _adapter_files in miners
            },
            completion_lens={
                miner_id: [0] * len(instances) for miner_id, _adapter_files in miners
            },
            search_queries={
                miner_id: [None] * len(instances) for miner_id, _adapter_files in miners
            },
            followup_requests=[],
            followup_budgets=[],
            followup_positions=[],
        )

        cursor = 0
        for miner_id, adapter_files in miners:
            for idx, (instance, original, prompt) in enumerate(
                zip(instances, originals, prompts)
            ):
                first_response, _tokens = first_completions[cursor]
                cursor += 1
                first_len = self._inference.count_tokens(
                    strip_untracked_spans(first_response)
                )
                search_query = parse_search_query(first_response)
                state.search_queries[miner_id][idx] = search_query
                if search_query is None:
                    state.final_answers[miner_id][idx] = extract_final_answer(
                        first_response
                    )
                    state.completion_lens[miner_id][idx] = first_len
                    continue

                state.followup_positions.append((miner_id, idx, first_len))
                state.followup_requests.append(
                    (
                        miner_id,
                        adapter_files,
                        self._build_search_followup_prompt(
                            prompt, first_response, search_query
                        ),
                    )
                )
                state.followup_budgets.append(
                    self._miner_budget(original, spent_tokens=first_len)
                )
        return state

    def _complete_batched_followups(self, state: _BatchedMinerAnswerState) -> None:
        if not state.followup_requests:
            return
        with self._progress(
            len(state.followup_requests), "miner long-context final answers"
        ) as progress:
            final_completions = self._inference.generate_for_miners_batch(
                state.followup_requests,
                max_new_tokens_list=state.followup_budgets,
                system_prompt=MINER_SYSTEM_PROMPT,
            )
            progress.update(len(state.followup_requests))
        if len(final_completions) != len(state.followup_requests):
            raise ValueError("miner must return exactly one final response per followup")
        for (miner_id, idx, first_len), (final_response, _tokens) in zip(
            state.followup_positions, final_completions
        ):
            final_len = self._inference.count_tokens(
                strip_untracked_spans(final_response)
            )
            state.final_answers[miner_id][idx] = extract_final_answer(final_response)
            state.completion_lens[miner_id][idx] = first_len + final_len

    def _answers_by_miner(
        self,
        miners: list[tuple[str, dict[str, bytes]]],
        instances: list[LongContextQAInstance],
        state: _BatchedMinerAnswerState,
    ) -> dict[str, list[LongContextAnswer]]:
        answers_by_miner: dict[str, list[LongContextAnswer]] = {}
        for miner_id, _adapter_files in miners:
            miner_answers: list[LongContextAnswer] = []
            for idx, instance in enumerate(instances):
                answer = str(state.final_answers[miner_id][idx] or "")
                chosen = parse_mc_choice(answer, instance.options)
                text = chosen if chosen is not None else answer
                miner_answers.append(
                    LongContextAnswer(
                        text=text,
                        completion_len=state.completion_lens[miner_id][idx],
                        verified=text == instance.gold_answer,
                    )
                )
            answers_by_miner[miner_id] = miner_answers
        return answers_by_miner

    def _score_batched_answers(
        self,
        *,
        miners: list[tuple[str, dict[str, bytes]]],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
        answers_by_miner: dict[str, list[LongContextAnswer]],
        search_queries: dict[str, list[str | None]],
    ) -> dict[str, list[LongContextMinerResult]]:
        miner_ids = [miner_id for miner_id, _adapter_files in miners]
        rewards_by_miner = self._batched_answer_rewards(
            miner_ids, instances, originals, answers_by_miner
        )
        return {
            miner_id: [
                LongContextMinerResult(
                    score=score,
                    original=original,
                    miner=miner,
                    search_query=search_query,
                )
                for original, miner, score, search_query in zip(
                    originals,
                    answers_by_miner[miner_id],
                    rewards_by_miner[miner_id],
                    search_queries[miner_id],
                )
            ]
            for miner_id in miner_ids
        }

    def _batched_answer_rewards(
        self,
        miner_ids: list[str],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer],
        answers_by_miner: dict[str, list[LongContextAnswer]],
    ) -> dict[str, list[float]]:
        rewards_by_miner: dict[str, list[float]] = {
            miner_id: [0.0] * len(instances) for miner_id in miner_ids
        }
        for idx, (instance, original) in enumerate(zip(instances, originals)):
            base_rewards = [
                self._mc_reward(
                    original,
                    answers_by_miner[miner_id][idx].completion_len,
                    answers_by_miner[miner_id][idx].text,
                    instance.gold_answer,
                )
                for miner_id in miner_ids
            ]
            rewards = peer_completion_efficiency_rewards(
                original_verified=original.verified,
                miner_verified=[
                    answers_by_miner[miner_id][idx].verified
                    for miner_id in miner_ids
                ],
                miner_completion_lens=[
                    answers_by_miner[miner_id][idx].completion_len
                    for miner_id in miner_ids
                ],
                base_rewards=base_rewards,
            )
            for miner_id, reward in zip(miner_ids, rewards):
                rewards_by_miner[miner_id][idx] = reward
        return rewards_by_miner

    @staticmethod
    def _mc_reward(
        original: LongContextAnswer,
        miner_completion_len: int,
        chosen_text: str,
        gold_answer: str,
    ) -> float:
        if chosen_text != gold_answer:
            return -1.0
        return relative_reasoning_reward(
            original_verified=original.verified,
            miner_verified=True,
            original_completion_len=original.completion_len,
            miner_completion_len=miner_completion_len,
        )

    def score_original_batch(
        self, instances: list[LongContextQAInstance]
    ) -> list[LongContextAnswer]:
        if not instances:
            return []
        prompts: list[str] = []
        for instance in instances:
            hits = self._retriever.search(instance.question, topk=self._config.answer_context_topk)
            prompts.append(self._build_original_mc_prompt(instance.question, hits, instance.options))

        completions = self._generate_original(
            prompts,
            max_new_tokens=self._config.original_answer_max_new_tokens,
            enable_thinking=True,
            progress_label="baseline long-context QA",
        )
        if len(completions) != len(instances):
            raise ValueError("original model must return exactly one baseline answer per question")

        results: list[LongContextAnswer] = []
        for instance, (answer, _tokens) in zip(instances, completions):
            completion_len = self._inference.count_tokens(strip_untracked_spans(answer))
            chosen = parse_mc_choice(answer, instance.options)
            text = chosen if chosen is not None else answer.strip()
            results.append(
                LongContextAnswer(
                    text=text,
                    completion_len=completion_len,
                    verified=text == instance.gold_answer,
                )
            )
        return results

    def _miner_budget(
        self,
        original: LongContextAnswer,
        *,
        spent_tokens: int = 0,
    ) -> int:
        budget = (
            int(original.completion_len)
            + max(0, int(self._config.miner_search_extra_tokens))
            - max(0, int(spent_tokens))
        )
        return max(1, budget)

    def _build_search_followup_prompt(
        self,
        prompt: str,
        first_response: str,
        search_query: str,
    ) -> str:
        hits = self._retriever.search(
            search_query, topk=self._config.answer_context_topk
        )
        information = format_hits(
            hits, max_chars_per_doc=self._config.max_chars_per_doc
        )
        information_block = f"<information>{information}</information>"
        response_with_information = inject_information_after_search(
            first_response.strip(),
            information_block,
        )
        return (
            f"{prompt}\n\n"
            f"{response_with_information}\n\n"
            "Continue the same response after the injected information block. "
            "Do not repeat earlier text."
        )

    def _answer_with_miner_batch(
        self,
        miner_id: str,
        adapter_files: dict[str, bytes],
        instances: list[LongContextQAInstance],
        originals: list[LongContextAnswer] | None = None,
    ) -> list[tuple[LongContextAnswer, str | None]]:
        if not instances:
            return []
        if originals is None:
            originals = self.score_original_batch(instances)
        if len(originals) != len(instances):
            raise ValueError("originals must match instances")
        prompts = [
            self._build_miner_prompt(instance.question, instance.options) for instance in instances
        ]
        first_budgets = [self._miner_budget(original) for original in originals]
        first_completions = self._generate_miner_with_budgets(
            miner_id,
            adapter_files,
            prompts,
            first_budgets,
            stop=["</search>"],
            system_prompt=MINER_SYSTEM_PROMPT,
            progress_label=f"miner {miner_id} long-context first pass",
        )
        if len(first_completions) != len(instances):
            raise ValueError("miner must return exactly one first response per question")

        final_answers: list[str | None] = [None] * len(instances)
        completion_lens: list[int] = [0] * len(instances)
        search_queries: list[str | None] = [None] * len(instances)
        followup_indexes: list[int] = []
        followup_prompts: list[str] = []
        followup_first_lens: list[int] = []
        followup_budgets: list[int] = []

        for idx, (instance, original, prompt, (first_response, _tokens)) in enumerate(
            zip(instances, originals, prompts, first_completions)
        ):
            first_len = self._inference.count_tokens(strip_untracked_spans(first_response))
            search_query = parse_search_query(first_response)
            search_queries[idx] = search_query
            if search_query is None:
                final_answers[idx] = extract_final_answer(first_response)
                completion_lens[idx] = first_len
                continue

            followup_indexes.append(idx)
            followup_first_lens.append(first_len)
            followup_prompts.append(
                self._build_search_followup_prompt(
                    prompt, first_response, search_query
                )
            )
            followup_budgets.append(
                self._miner_budget(original, spent_tokens=first_len)
            )

        if followup_prompts:
            final_completions = self._generate_miner_with_budgets(
                miner_id,
                adapter_files,
                followup_prompts,
                followup_budgets,
                system_prompt=MINER_SYSTEM_PROMPT,
                progress_label=f"miner {miner_id} long-context final answers",
            )
            if len(final_completions) != len(followup_prompts):
                raise ValueError("miner must return exactly one final response per followup")
            for idx, first_len, (final_response, _tokens) in zip(
                followup_indexes, followup_first_lens, final_completions
            ):
                final_len = self._inference.count_tokens(strip_untracked_spans(final_response))
                final_answers[idx] = extract_final_answer(final_response)
                completion_lens[idx] = first_len + final_len

        results: list[tuple[LongContextAnswer, str | None]] = []
        for instance, answer, completion_len, search_query in zip(
            instances, final_answers, completion_lens, search_queries
        ):
            chosen = parse_mc_choice(str(answer or ""), instance.options)
            text = chosen if chosen is not None else str(answer or "")
            results.append(
                (
                    LongContextAnswer(
                        text=text,
                        completion_len=completion_len,
                        verified=text == instance.gold_answer,
                    ),
                    search_query,
                )
            )
        return results

    def _judge_answer(self, question: str, gold_answer: str, candidate_answer: str) -> bool:
        return self._judge_answers_batch([(question, gold_answer, candidate_answer)])[0]

    def _judge_answers_batch(self, items: list[tuple[str, str, str]]) -> list[bool]:
        """Used only to filter sampled distractors against the gold answer
        when building multiple-choice options (see `_generate_distractors_batch`).
        Both sides of this comparison come from the validator's own model, so
        it is not exposed to miner-controlled input.
        """
        if not items:
            return []
        prompts = [
            self._build_judge_prompt(question, gold_answer, candidate_answer)
            for question, gold_answer, candidate_answer in items
        ]
        completions = self._generate_original(
            prompts,
            max_new_tokens=self._config.judge_max_new_tokens,
            greedy=True,
            progress_label="long-context distractor validation",
        )
        if len(completions) != len(items):
            raise ValueError("judge must return exactly one response per candidate")
        return [parse_judgement(completion) for completion, _tokens in completions]

    def _build_judge_prompt(self, question: str, gold_answer: str, candidate_answer: str) -> str:
        return (
            "You are judging whether a candidate answer is semantically equivalent to "
            "the gold answer for a closed-world QA task. Allow paraphrases, aliases, "
            "unit-equivalent numbers, and different wording. Reject unsupported or "
            "contradictory answers.\n\n"
            f"Question:\n{question}\n\n"
            f"Gold answer:\n{gold_answer}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            'Return JSON only: {"equivalent": true} or {"equivalent": false}.'
        )

    def _build_qa_generation_prompt(self, seed_hits: tuple[RetrievalHit, ...]) -> str:
        context = format_hits(
            seed_hits,
            max_chars_per_doc=self._config.qa_generation_max_chars_per_doc,
        )
        return (
            "You are creating a difficult long-context QA challenge for a validator. "
            "Use only the documents below as ground truth.\n\n"
            "The question must be hard in two distinct ways:\n"
            "1. Hard to answer: it must require specific evidence from the documents "
            "and must not be answerable from broad common knowledge alone.\n"
            "2. Hard to retrieve: it must NOT reuse the distinctive words, names, or "
            "phrases from the passage that contains the answer, so a lexical search "
            "engine (e.g. BM25) cannot match it by keyword overlap; and it must not "
            "be a close paraphrase of that passage, so a semantic/embedding search "
            "cannot match it by similarity either. Prefer oblique references to "
            "entities (by role, relation, attribute, or description rather than by "
            "name), multi-hop reasoning that combines facts from different parts of "
            "the documents, or questions about implied/inferred details rather than "
            "directly quoted text.\n\n"
            "The answer must still be concise and fully grounded in the documents.\n\n"
            f"Documents:\n{context}\n\n"
            "Use this JSON format exactly. The content below is only an example; "
            "create a new question and answer from the documents above.\n\n"
            "```json\n"
            "{\n"
            '  "question": "Who described the Analytical Engine algorithm?",\n'
            '  "answer": "Ada Lovelace"\n'
            "}\n"
            "```\n\n"
            "Return exactly one JSON object and no explanation."
        )

    def _build_qa_repair_prompt(
        self,
        *,
        prompt: str,
        invalid_completion: str,
        error: Exception,
        seed_hits: tuple[RetrievalHit, ...],
    ) -> str:
        context = format_hits(
            seed_hits,
            max_chars_per_doc=self._config.qa_generation_max_chars_per_doc,
        )
        return (
            f"{prompt}\n\n"
            "The previous response was invalid and could not be parsed as the required "
            f"JSON object. Parse error: {error}\n\n"
            f"Previous invalid response:\n{invalid_completion.strip()}\n\n"
            "Return a replacement response using only this exact form, with no markdown "
            "fence, no prose, and no extra keys:\n"
            '{"question": "A concise question grounded in the documents", '
            '"answer": "A concise answer"}\n\n'
            f"Documents, repeated for grounding:\n{context}"
        )

    def _build_distractor_prompt(self, question: str) -> str:
        return (
            "Answer the question directly from your own knowledge, in a few words. "
            "If you are not sure, still give your single best guess. No explanation, "
            "no punctuation beyond what the answer itself needs.\n\n"
            f"Question:\n{question}\nAnswer:"
        )

    @staticmethod
    def _format_options(options: tuple[str, ...]) -> str:
        lines = []
        for index, option in enumerate(options):
            letter = chr(ord("A") + index)
            lines.append(f"{letter}) {option}")
        return "\n".join(lines)

    def _build_miner_prompt(self, question: str, options: tuple[str, ...]) -> str:
        return f"Question:\n{question}\n\nOptions:\n{self._format_options(options)}"

    def _build_original_mc_prompt(
        self, question: str, hits: list[RetrievalHit], options: tuple[str, ...]
    ) -> str:
        context = format_hits(hits, max_chars_per_doc=self._config.max_chars_per_doc)
        return (
            "Use the reference documents to answer the multiple-choice question. Do "
            "not use XML tags or search syntax. Respond with only the letter of the "
            "correct option.\n\n"
            f"Reference documents:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{self._format_options(options)}\n\n"
            "Answer with just the letter:"
        )


__all__ = [
    "LongContextAnswer",
    "LongContextInferenceBackend",
    "LongContextMinerResult",
    "LongContextQAConfig",
    "LongContextQAEvaluator",
    "LongContextQAInstance",
    "MINER_SYSTEM_PROMPT",
    "extract_final_answer",
    "inject_information_after_search",
    "parse_generated_qa",
    "parse_judgement",
    "parse_mc_choice",
    "parse_search_query",
    "strip_untracked_spans",
]
